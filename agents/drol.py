import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

from utils.typing import PRNGKey, FloatScalar, IntScalar
import jax.random as jr

def sample_uniform_in_hypersphere(key: PRNGKey, R: float, shape: tuple[IntScalar, ...]) -> jnp.ndarray:
    key_norm, key_u = jr.split(key)
    d = shape[-1]  # The last dimension is the dimension of the hypersphere

    v = jr.normal(key_norm, shape)
    norms = safe_norm(v, axis=-1, keepdims=True)
    directions = v / (norms + 1e-8)

    radius_shape = shape[:-1] + (1,)
    u = jr.uniform(key_u, radius_shape)
    radii = R * (u ** (1.0 / (d + 1e-8)))

    samples = radii * directions

    return samples

def safe_norm(x: jnp.ndarray, axis: int = None, keepdims: bool = False, eps: FloatScalar = 1e-6) -> jnp.ndarray:
    """Compute the norm of a vector, with a small epsilon to avoid division by zero."""
    return jnp.sqrt(jnp.sum(x ** 2, axis=axis, keepdims=keepdims) + eps)

class DROLAgent(flax.struct.PyTreeNode):
    """Minimal hard-best FQL variant with K-candidate matching."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _sample_actions_single(self, observations, seed):
        """Sample one action per state from the one-step flow policy."""
        action_dim = self.config['action_dim']
        r_max = jnp.sqrt(jnp.asarray(action_dim, dtype=jnp.float32))
        noises = sample_uniform_in_hypersphere(
            seed,
            R=r_max,
            shape=(
                *observations.shape[: -len(self.config['ob_dims'])],
                action_dim,
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        return jnp.clip(actions, -1, 1)

    def _sample_action_candidates(self, observations, seed, num_candidates):
        """Sample multiple one-step flow actions per state."""
        action_dim = self.config['action_dim']
        r_max = jnp.sqrt(jnp.asarray(action_dim, dtype=jnp.float32))
        batch_size = observations.shape[0]

        noises = sample_uniform_in_hypersphere(
            seed,
            R=r_max,
            shape=(batch_size, num_candidates, action_dim),
        )
        obs_expanded = jnp.repeat(observations[:, None, ...], num_candidates, axis=1)
        obs_flat = obs_expanded.reshape((-1, *observations.shape[1:]))
        noises_flat = noises.reshape((-1, action_dim))

        candidate_actions = self.network.select('actor_onestep_flow')(obs_flat, noises_flat)
        candidate_actions = candidate_actions.reshape((batch_size, num_candidates, action_dim))
        return jnp.clip(candidate_actions, -1, 1)

    def _select_candidate_actions_with_critic(self, observations, candidate_actions, *, critic_module, q_agg):
        """Rank candidate actions with a critic and return the best one per state."""
        num_candidates = candidate_actions.shape[1]
        action_dim = self.config['action_dim']
        flat_actions = candidate_actions.reshape((-1, action_dim))
        flat_obs = jnp.repeat(observations, num_candidates, axis=0)

        qs = self.network.select(critic_module)(flat_obs, actions=flat_actions)
        if q_agg == 'min':
            q_flat = qs.min(axis=0)
        else:
            q_flat = qs.mean(axis=0)
        q_scores = q_flat.reshape((observations.shape[0], num_candidates))
        best_idx = jnp.argmax(q_scores, axis=1)
        best_actions = jnp.take_along_axis(candidate_actions, best_idx[:, None, None], axis=1).squeeze(axis=1)
        return best_actions

    def _sample_backup_actions(self, observations, seed, num_candidates):
        """Sample next actions for TD backup, optionally with conservative best-of-N selection."""
        if num_candidates <= 1:
            return self._sample_actions_single(observations, seed=seed)

        candidate_actions = self._sample_action_candidates(observations, seed, num_candidates)
        return self._select_candidate_actions_with_critic(
            observations,
            candidate_actions,
            critic_module='target_critic',
            q_agg=self.config['critic_selection_q_agg'],
        )

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic TD loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self._sample_backup_actions(
            batch['next_observations'],
            seed=sample_rng,
            num_candidates=int(self.config['critic_num_candidates']),
        )

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute minimal hard-best actor loss over K candidates."""
        batch_size, action_dim = batch['actions'].shape
        k = self.config['num_candidates']
        r_max = jnp.sqrt(jnp.asarray(action_dim, dtype=jnp.float32))

        rng, noise_rng = jax.random.split(rng)

        noises = sample_uniform_in_hypersphere(noise_rng, R=r_max, shape=(batch_size, k, action_dim))
        obs_expanded = jnp.repeat(batch['observations'][:, None, ...], k, axis=1)

        obs_flat = obs_expanded.reshape((-1, *batch['observations'].shape[1:]))
        noises_flat = noises.reshape((-1, action_dim))

        candidate_actions = self.network.select('actor_onestep_flow')(obs_flat, noises_flat, params=grad_params)
        candidate_actions = candidate_actions.reshape((batch_size, k, action_dim))
        candidate_actions = jnp.clip(candidate_actions, -1, 1)

        action_diffs = candidate_actions - batch['actions'][:, None, :]
        candidate_d2 = jnp.sum(action_diffs**2, axis=-1)
        best_idx = jnp.argmin(candidate_d2, axis=1)
        diffs = candidate_actions[:, :, None, :] - candidate_actions[:, None, :, :]
        pairwise_dist2 = jnp.sum(diffs**2, axis=-1)
        eye = jnp.eye(k, dtype=pairwise_dist2.dtype)[None, :, :]
        valid_pairs = 1.0 - eye
        action_dim_f = jnp.maximum(jnp.asarray(action_dim, dtype=pairwise_dist2.dtype), 1.0)
        candidate_d2_per_dim = candidate_d2 / action_dim_f
        pairwise_dist2_per_dim = pairwise_dist2 / action_dim_f

        # Hard-best BC: optimize nearest candidate with per-dimension normalization.
        bc_loss = jnp.take_along_axis(candidate_d2_per_dim, best_idx[:, None], axis=1).mean()

        lambda_val = 0.1  # 强烈建议固定在这个值
        selected_d2 = jnp.take_along_axis(candidate_d2_per_dim, best_idx[:, None], axis=1).squeeze(axis=1)
        gate = jnp.exp(-selected_d2 / lambda_val)
        gate = jax.lax.stop_gradient(gate)  # 【极其重要】绝对不能让 Q 的梯度从这里传回去！

        qs = self.network.select('critic')(obs_flat, actions=candidate_actions.reshape((-1, action_dim)))
        if self.config['actor_q_agg'] == 'min':
            q_flat = qs.min(axis=0)
        else:
            q_flat = qs.mean(axis=0)
        q_candidates = q_flat.reshape((batch_size, k))

        # Hard-best DDPG: only optimize Q for the nearest candidate.
        q_selected_per_state = jnp.take_along_axis(q_candidates, best_idx[:, None], axis=1).squeeze(axis=1)

        # q_loss = - (gate * q_selected_per_state).mean()
        q_loss = - q_selected_per_state.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.maximum(jnp.abs(q_selected_per_state).mean(), 1e-6))
            q_loss = lam * q_loss

        actor_loss = self.config['bc_coef'] * bc_loss + self.config['q_coef'] * q_loss

        min_d2 = candidate_d2_per_dim.min(axis=1)
        candidate_center = candidate_actions.mean(axis=1, keepdims=True)
        candidate_divergence = jnp.mean(jnp.sum((candidate_actions - candidate_center) ** 2, axis=-1) / action_dim_f)
        per_sample_pairwise_mean_dist2 = jnp.sum(pairwise_dist2_per_dim * valid_pairs, axis=(1, 2)) / jnp.maximum(
            valid_pairs.sum(), 1.0
        )
        pairwise_mean_dist2 = per_sample_pairwise_mean_dist2.mean()
        pairwise_mean_dist = jnp.sqrt(jnp.maximum(pairwise_mean_dist2, 0.0))
        best_idx_probs = jax.nn.one_hot(best_idx, k).mean(axis=0)
        best_idx_entropy = -jnp.sum(best_idx_probs * jnp.log(jnp.maximum(best_idx_probs, 1e-8)))
        best_idx_max_prob = best_idx_probs.max()

        actions = self._sample_actions_single(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'q_loss': q_loss,
            'q': q_selected_per_state.mean(),
            'mse': mse,
            'gate': gate.mean(),
            'min_d2': min_d2.mean(),
            'candidate_divergence': candidate_divergence,
            'candidate_pairwise_dist2': pairwise_mean_dist2,
            'candidate_pairwise_dist': pairwise_mean_dist,
            'best_idx_entropy': best_idx_entropy,
            'best_idx_max_prob': best_idx_max_prob,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute total critic + actor loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update target critic parameters with EMA."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with metrics."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions for inference.

        Default behavior is single-sample one-step flow sampling.
        When enabled, uses Best-of-N with density filtering:
        1) sample N candidates,
        2) keep dense top-K by pairwise distances,
        3) pick highest-Q action within dense candidates.
        """
        del temperature
        seed = self.rng if seed is None else seed

        if (not self.config['inference_best_of_n']) or (self.config['inference_num_samples'] <= 1):
            return self._sample_actions_single(observations, seed=seed)

        added_batch_dim = False
        if observations.ndim == len(self.config['ob_dims']):
            observations = observations[None, ...]
            added_batch_dim = True

        batch_size = observations.shape[0]
        num_samples = self.config['inference_num_samples']
        candidate_actions = self._sample_action_candidates(observations, seed, num_samples)

        selected_actions = candidate_actions
        if self.config['inference_use_density_filter'] and num_samples > 1:
            diffs = candidate_actions[:, :, None, :] - candidate_actions[:, None, :, :]
            pairwise_dist2 = jnp.sum(diffs**2, axis=-1)
            eye = jnp.eye(num_samples, dtype=pairwise_dist2.dtype)[None, :, :]
            valid = 1.0 - eye
            mean_dist2 = jnp.sum(pairwise_dist2 * valid, axis=-1) / jnp.maximum(valid.sum(axis=-1), 1.0)

            dense_topk = min(int(self.config['inference_dense_topk']), int(num_samples))
            dense_topk = max(dense_topk, 1)
            dense_idx = jnp.argsort(mean_dist2, axis=1)[:, :dense_topk]
            selected_actions = jnp.take_along_axis(candidate_actions, dense_idx[:, :, None], axis=1)

        best_actions = self._select_candidate_actions_with_critic(
            observations,
            selected_actions,
            critic_module='critic',
            q_agg=self.config['inference_q_agg'],
        )

        if added_batch_dim:
            best_actions = best_actions[0]
        return best_actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent instance."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='drol',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation for target Q in critic TD backup.
            actor_q_agg='min',  # Aggregation for actor objective ('min' is conservative).
            num_candidates=16,  # Number of latent-action candidates per state.
            critic_num_candidates=1,  # Number of next-action candidates for TD backup; >1 enables backup best-of-N.
            critic_selection_q_agg='min',  # Aggregation used to rank backup candidates with target critic.
            bc_coef=1.0,  # Behavior matching coefficient.
            q_coef=1.0,  # Q maximization coefficient.
            normalize_q_loss=True,  # Whether to normalize the Q loss scale.
            inference_best_of_n=False,  # Enable Best-of-N sampling only during inference.
            inference_num_samples=32,  # Number of candidate actions for inference Best-of-N.
            inference_use_density_filter=False,  # Filter isolated actions via pairwise-distance density.
            inference_dense_topk=8,  # Keep top-K densest candidates before Q-based selection.
            inference_q_agg='mean',  # Critic ensemble aggregation for inference action ranking.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
