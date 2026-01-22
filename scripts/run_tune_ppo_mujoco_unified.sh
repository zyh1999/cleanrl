#!/usr/bin/env bash
#
# CleanRL PPO (continuous action) - 多 MuJoCo 环境“统一调参”（同一套超参）
#
# 覆盖的环境列表（与你的 run_ppo_all_mujoco_parallel.sh 一致）：
#   HalfCheetah-v4, Hopper-v4, Walker2d-v4, Swimmer-v4, Humanoid-v4
#
# 用法：
#   cd /home/yihe/cleanrl
#   # 首次需要安装 optuna 依赖（按 CleanRL 文档）
#   # uv pip install ".[optuna]"
#   bash scripts/run_tune_ppo_mujoco_unified.sh
#
# 可选环境变量：
#   NUM_TRIALS=50
#   NUM_SEEDS=2
#   TOTAL_TIMESTEPS=500000
#   METRIC_LAST_N=20
#   AGG=average                # average/median/min
#   STORAGE="sqlite:///cleanrl_mujoco_unified_hpopt.db"
#   STUDY_NAME="mujoco_unified_v1"
#   NORM_REWARD=False          # True/False（会传 --no-norm-reward）
#   TRACK_WANDB=False          # True/False（只记录 trial 聚合分数，不刷每个 env/seed 的 run）
#   WANDB_PROJECT="cleanrl_mujoco_unified_tune"
#   WANDB_ENTITY=""
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

NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_SEEDS="${NUM_SEEDS:-2}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
METRIC_LAST_N="${METRIC_LAST_N:-20}"
AGG="${AGG:-average}"
STORAGE="${STORAGE:-sqlite:///cleanrl_mujoco_unified_hpopt.db}"
STUDY_NAME="${STUDY_NAME:-}"
NORM_REWARD="${NORM_REWARD:-False}"
TRACK_WANDB="${TRACK_WANDB:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-cleanrl_mujoco_unified_tune}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

extra_args=()
if [[ -n "${STUDY_NAME}" ]]; then
  extra_args+=(--study-name "${STUDY_NAME}")
fi

if [[ "${TRACK_WANDB}" == "True" || "${TRACK_WANDB}" == "true" || "${TRACK_WANDB}" == "1" ]]; then
  extra_args+=(--track-wandb --wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    extra_args+=(--wandb-entity "${WANDB_ENTITY}")
  fi
fi

# norm_reward flag
if [[ "${NORM_REWARD}" == "True" || "${NORM_REWARD}" == "true" || "${NORM_REWARD}" == "1" ]]; then
  extra_args+=(--norm-reward)
else
  extra_args+=(--no-norm-reward)
fi

echo "========================================================"
echo "CleanRL unified MuJoCo tuning"
echo "NUM_TRIALS=${NUM_TRIALS} | NUM_SEEDS=${NUM_SEEDS} | TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "AGG=${AGG} | METRIC_LAST_N=${METRIC_LAST_N}"
echo "STORAGE=${STORAGE} | STUDY_NAME=${STUDY_NAME:-<auto>}"
echo "NORM_REWARD=${NORM_REWARD}"
echo "TRACK_WANDB=${TRACK_WANDB} | WANDB_PROJECT=${WANDB_PROJECT} | WANDB_ENTITY=${WANDB_ENTITY:-<none>}"
echo "========================================================"

python scripts/tune_ppo_mujoco_unified.py \
  --num-trials "${NUM_TRIALS}" \
  --num-seeds "${NUM_SEEDS}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --metric-last-n "${METRIC_LAST_N}" \
  --aggregation "${AGG}" \
  --storage "${STORAGE}" \
  "${extra_args[@]}"

