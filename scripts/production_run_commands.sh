#!/usr/bin/env bash

set -euo pipefail

# Run this script from /Users/muzhancun/workspace/fql/dorl.
# It collects the production commands used to reproduce the DROL runs.
#
# Sections:
# 1. OGBench DROL(16): fixed default K=16.
# 2. OGBench DROL*: family-level tuned K chosen once on the representative task.
# 3. D4RL DROL(16): fixed default K=16.
# 4. D4RL tuned supporting reruns.
#
# The OGBench bc_coef values follow the production settings in wbrl_paper.txt and
# sweep_num_candidates_production*.txt.
# The OGBench tuned K values follow num_candidates_widest_taskgroup_results.txt.
# For D4RL, the default bc_coef values follow the D4RL sweeps; the tuned section
# keeps the widest-sweep/shared-K choices where available and uses bc_coef=0.03 for
# antmaze-large-play/large-diverse, matching the production best-vs-K16 reruns.

run_ogbench_family() {
  local family="$1"
  local bc_coef="$2"
  local num_candidates="$3"
  shift 3
  local extra_flags=("$@")

  for task in 1 2 3 4 5; do
    python main.py \
      --env_name="${family}-task${task}-v0" \
      --agent=agents/dorl.py \
      --run_group=production \
      --agent.bc_coef="${bc_coef}" \
      --agent.num_candidates="${num_candidates}" \
      "${extra_flags[@]}"
  done
}

run_d4rl_env() {
  local env_name="$1"
  local bc_coef="$2"
  local num_candidates="$3"
  shift 3
  local extra_flags=("$@")

  python main.py \
    --env_name="${env_name}" \
    --agent=agents/dorl.py \
    --offline_steps=500000 \
    --run_group=production \
    --agent.bc_coef="${bc_coef}" \
    --agent.num_candidates="${num_candidates}" \
    "${extra_flags[@]}"
}

echo "# OGBench DROL(16)"
run_ogbench_family antmaze-giant-navigate-singletask 0.03 16 --agent.discount=0.995 --agent.q_agg=min
run_ogbench_family antmaze-large-navigate-singletask 0.03 16 --agent.discount=0.995 --agent.q_agg=min
run_ogbench_family antsoccer-arena-navigate-singletask 0.1 16 --agent.discount=0.995
run_ogbench_family humanoidmaze-medium-navigate-singletask 0.3 16 --agent.discount=0.995
run_ogbench_family humanoidmaze-large-navigate-singletask 0.3 16 --agent.discount=0.995
run_ogbench_family cube-single-play-singletask 10.0 16
run_ogbench_family cube-double-play-singletask 3.0 16
run_ogbench_family scene-play-singletask 3.0 16
run_ogbench_family puzzle-3x3-play-singletask 0.3 16
run_ogbench_family puzzle-4x4-play-singletask 10.0 16

echo "# OGBench DROL*"
run_ogbench_family antmaze-giant-navigate-singletask 0.03 1 --agent.discount=0.995 --agent.q_agg=min
run_ogbench_family antmaze-large-navigate-singletask 0.03 16 --agent.discount=0.995 --agent.q_agg=min
run_ogbench_family antsoccer-arena-navigate-singletask 0.1 4 --agent.discount=0.995
run_ogbench_family humanoidmaze-medium-navigate-singletask 0.3 16 --agent.discount=0.995
run_ogbench_family humanoidmaze-large-navigate-singletask 0.3 4 --agent.discount=0.995
run_ogbench_family cube-single-play-singletask 10.0 16
run_ogbench_family cube-double-play-singletask 3.0 64
run_ogbench_family scene-play-singletask 3.0 64
run_ogbench_family puzzle-3x3-play-singletask 0.3 32
run_ogbench_family puzzle-4x4-play-singletask 10.0 64

echo "# D4RL DROL(16)"
run_d4rl_env antmaze-umaze-v2 0.3 16
run_d4rl_env antmaze-umaze-diverse-v2 0.1 16
run_d4rl_env antmaze-medium-play-v2 0.1 16
run_d4rl_env antmaze-medium-diverse-v2 0.1 16
run_d4rl_env antmaze-large-play-v2 0.1 16
run_d4rl_env antmaze-large-diverse-v2 0.1 16

run_d4rl_env pen-human-v1 1.0 16 --agent.q_agg=min
run_d4rl_env pen-cloned-v1 1.0 16 --agent.q_agg=min
run_d4rl_env pen-expert-v1 1.0 16 --agent.q_agg=min

run_d4rl_env door-human-v1 10.0 16 --agent.q_agg=min
run_d4rl_env door-cloned-v1 10.0 16 --agent.q_agg=min
run_d4rl_env door-expert-v1 10.0 16 --agent.q_agg=min

run_d4rl_env hammer-human-v1 3.0 16 --agent.q_agg=min
run_d4rl_env hammer-cloned-v1 3.0 16 --agent.q_agg=min
run_d4rl_env hammer-expert-v1 3.0 16 --agent.q_agg=min

run_d4rl_env relocate-human-v1 10.0 16 --agent.q_agg=min
run_d4rl_env relocate-cloned-v1 10.0 16 --agent.q_agg=min
run_d4rl_env relocate-expert-v1 10.0 16 --agent.q_agg=min

echo "# D4RL tuned supporting reruns"
run_d4rl_env antmaze-umaze-v2 0.3 2
run_d4rl_env antmaze-umaze-diverse-v2 0.1 16
run_d4rl_env antmaze-medium-play-v2 0.1 32
run_d4rl_env antmaze-medium-diverse-v2 0.1 16
run_d4rl_env antmaze-large-play-v2 0.03 16
run_d4rl_env antmaze-large-diverse-v2 0.03 16

run_d4rl_env pen-human-v1 1.0 16 --agent.q_agg=min
run_d4rl_env pen-cloned-v1 1.0 16 --agent.q_agg=min
run_d4rl_env pen-expert-v1 1.0 16 --agent.q_agg=min

run_d4rl_env door-human-v1 10.0 8 --agent.q_agg=min
run_d4rl_env door-cloned-v1 10.0 8 --agent.q_agg=min
run_d4rl_env door-expert-v1 10.0 8 --agent.q_agg=min

run_d4rl_env hammer-human-v1 3.0 8 --agent.q_agg=min
run_d4rl_env hammer-cloned-v1 3.0 8 --agent.q_agg=min
run_d4rl_env hammer-expert-v1 3.0 8 --agent.q_agg=min

run_d4rl_env relocate-human-v1 10.0 32 --agent.q_agg=min
run_d4rl_env relocate-cloned-v1 10.0 32 --agent.q_agg=min
run_d4rl_env relocate-expert-v1 10.0 32 --agent.q_agg=min
