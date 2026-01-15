#!/usr/bin/env bash
#
# CleanRL PPO (continuous action) - MuJoCo parallel launcher
#
# 目标：
# - 多个 MuJoCo 环境 + 多个 seed 并行跑
# - GPU 轮询分配（CUDA_VISIBLE_DEVICES）
# - Ctrl+C 一键终止当前批次所有子进程
# - stdout/stderr 落到 logs/ 便于排查
#
# 用法：
#   cd /home/yihe/cleanrl
#   bash scripys/run_ppo_all_mujoco_parallel.sh
#
# 常用覆盖（环境变量）：
#   GPU_COUNT=1 ENV_CONCURRENCY=2 TOTAL_TIMESTEPS=5000000 WANDB_PROJECT=cleanRL_mujoco bash scripts/run_ppo_all_mujoco_parallel.sh
#

set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

# repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/ppo_mujoco_parallel

# 并发参数
GPU_COUNT="${GPU_COUNT:-2}"                 # GPU 张数；会做轮询分配
ENV_CONCURRENCY="${ENV_CONCURRENCY:-2}"     # 每批并行跑几个环境（每个环境仍会跑所有 seeds）

# 训练超参：只保留 total timesteps（其余全部用 cleanrl/ppo_continuous_action.py 默认值）
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-5000000}"

# W&B
TRACK="${TRACK:-True}"                      # True/False
WANDB_PROJECT="${WANDB_PROJECT:-cleanRL_mujoco_ppo_parallel}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
# exp_name 会进入 run_name: {env_id}__{exp_name}__{seed}__{time}
EXP_NAME="${EXP_NAME:-ppo_continuous_mujoco_parallel}"

seeds=(9 1 2 0)
mujoco_envs=(
  "HalfCheetah-v4"
  "Hopper-v4"
  "Walker2d-v4"
  "Swimmer-v4"
  "Humanoid-v4"
)

pids=()
trap 'echo "Caught Ctrl+C, killing all runs..."; \
      for pid in "${pids[@]}"; do \
        kill "$pid" 2>/dev/null || true; \
      done; \
      wait || true; \
      echo "All runs killed."; \
      exit 1' INT

echo "Starting CleanRL PPO MuJoCo parallel runs"
echo "Envs: ${mujoco_envs[*]}"
echo "Seeds: ${seeds[*]}"
echo "GPU_COUNT=${GPU_COUNT}, ENV_CONCURRENCY=${ENV_CONCURRENCY}"
echo "TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "W&B: TRACK=${TRACK}, PROJECT=${WANDB_PROJECT}, ENTITY=${WANDB_ENTITY:-<none>}"

for ((i=0; i<${#mujoco_envs[@]}; i+=ENV_CONCURRENCY)); do
  batch_envs=("${mujoco_envs[@]:i:ENV_CONCURRENCY}")
  pids=()

  echo "========================================================"
  echo "Starting Env batch: ${batch_envs[*]}"
  echo "========================================================"

  for e_idx in "${!batch_envs[@]}"; do
    env_id="${batch_envs[$e_idx]}"
    log_dir="logs/ppo_mujoco_parallel/${env_id}"
    mkdir -p "${log_dir}"

    for s_idx in "${!seeds[@]}"; do
      seed="${seeds[$s_idx]}"

      global_job_id=$(( e_idx * ${#seeds[@]} + s_idx ))
      gpu=$(( global_job_id % GPU_COUNT ))

      log_file="${log_dir}/seed${seed}.log"
      echo "  -> Launching env=${env_id}, seed=${seed}, gpu=${gpu} (log: ${log_file})"

      # 组装参数（tyro 支持 --flag value 形式；bool 用 True/False）
      ARGS=(
        "--env-id" "${env_id}"
        "--seed" "${seed}"
        "--exp-name" "${EXP_NAME}"
        "--total-timesteps" "${TOTAL_TIMESTEPS}"
      )

      if [[ "${TRACK}" == "True" || "${TRACK}" == "true" || "${TRACK}" == "1" ]]; then
        ARGS+=("--track" "--wandb-project-name" "${WANDB_PROJECT}")
        if [[ -n "${WANDB_ENTITY}" ]]; then
          ARGS+=("--wandb-entity" "${WANDB_ENTITY}")
        fi
      fi

      CUDA_VISIBLE_DEVICES="${gpu}" python cleanrl/ppo_continuous_action.py \
        "${ARGS[@]}" \
        > "${log_file}" 2>&1 &

      pids+=($!)
    done
  done

  echo "Waiting for env batch to finish: ${batch_envs[*]}"
  wait "${pids[@]}" || echo "[WARN] Some runs exited with non-zero status in env batch: ${batch_envs[*]}. Check logs/ppo_mujoco_parallel/."
  echo "Env batch finished: ${batch_envs[*]}"
done

echo "All CleanRL PPO MuJoCo runs finished."


