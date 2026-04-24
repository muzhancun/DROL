import d4rl
import gymnasium
import h5py
import numpy as np
from pathlib import Path
import re

from envs.env_utils import EpisodeMonitor
from utils.datasets import Dataset


_MIXED_ENV_RE = re.compile(r'^(?P<task>.+)-human-expert-v(?P<version>\d+)$')


def _parse_mixed_env_name(env_name: str):
    return _MIXED_ENV_RE.match(env_name)


def _resolve_make_env_name(env_name: str) -> str:
    """Map mixed dataset names to a real D4RL env id for env creation."""
    match = _parse_mixed_env_name(env_name)
    if match is None:
        return env_name
    return f"{match.group('task')}-expert-v{match.group('version')}"


def _get_mixed_dataset_path(env_name: str) -> Path:
    """Resolve local path for a mixed dataset file."""
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / 'data' / 'd4rl_mixed' / f'{env_name}.hdf5',
        root / 'data' / f'{env_name}.hdf5',
        root / f'{env_name}.hdf5',
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f'Mixed dataset file for {env_name} not found. Tried: ' + ', '.join(str(p) for p in candidates)
    )


def _load_mixed_dataset(env_name: str):
    """Load mixed dataset from local HDF5 file."""
    path = _get_mixed_dataset_path(env_name)
    with h5py.File(path, 'r') as f:
        required_keys = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
        missing = [k for k in required_keys if k not in f]
        if missing:
            raise KeyError(f'Missing keys in {path}: {missing}')
        return {
            'observations': f['observations'][:],
            'actions': f['actions'][:],
            'next_observations': f['next_observations'][:],
            'rewards': f['rewards'][:],
            'terminals': f['terminals'][:],
        }


def make_env(env_name):
    """Make D4RL environment."""
    env_id = _resolve_make_env_name(env_name)
    env = gymnasium.make('GymV21Environment-v0', env_id=env_id)
    env = EpisodeMonitor(env)
    return env


def get_dataset(
    env,
    env_name,
    chunk_size: int = 5,
):
    """Make D4RL dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
        chunk_size: Number of consecutive actions per sample for chunked actions.
    """
    if _parse_mixed_env_name(env_name) is not None:
        dataset = _load_mixed_dataset(env_name)
    else:
        dataset = d4rl.qlearning_dataset(env)

    terminals = np.zeros_like(dataset['rewards'])  # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards'])  # Indicate whether we should bootstrap from the next state.
    rewards = dataset['rewards'].copy().astype(np.float32)
    if 'antmaze' in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
            )
            masks[i] = 1 - dataset['terminals'][i]
        rewards = rewards - 1.0
    else:
        for i in range(len(terminals) - 1):
            if (
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
                or dataset['terminals'][i] == 1.0
            ):
                terminals[i] = 1
            else:
                terminals[i] = 0
            masks[i] = 1 - dataset['terminals'][i]
    masks[-1] = 1 - dataset['terminals'][-1]
    terminals[-1] = 1

    actions = dataset['actions'].astype(np.float32)

    return Dataset.create(
        observations=dataset['observations'].astype(np.float32),
        actions=actions,
        next_observations=dataset['next_observations'].astype(np.float32),
        terminals=terminals.astype(np.float32),
        rewards=rewards,
        masks=masks,
        chunk_size=chunk_size,
    )
