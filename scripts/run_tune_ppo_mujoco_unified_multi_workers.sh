#!/usr/bin/env bash
#
# CleanRL unified MuJoCo tuning - 多 worker 并行版（一次跑多根）
#
# 原理：
# - 启动 N_WORKERS 个独立进程
# - 所有进程共享同一个 Optuna STORAGE + STUDY_NAME
# - Optuna 会把不同的 trial 分配给不同 worker，从而实现并行 trial
#
# 用法：
#   cd /home/yihe/cleanrl
#   bash scripts/run_tune_ppo_mujoco_unified_multi_workers.sh
#
# 可选环境变量：
#   N_WORKERS=4
#   GPU_IDS="0 1"             # 按 worker 轮询分配 CUDA_VISIBLE_DEVICES
#   TOTAL_TRIALS=40           # 固定总 trials；会自动均分到各 worker（不能整除时前几个 worker 多 1）
#   NUM_SEEDS=2
#   TOTAL_TIMESTEPS=1000000
#   METRIC_LAST_N=20
#   AGG=average               # average/median/min
#   STORAGE="sqlite:///cleanrl_mujoco_unified_hpopt.db"
#   STUDY_NAME="mujoco_unified_v1"   # 强烈建议显式指定，方便断点续跑/并行
#   TRACK_WANDB=False         # True/False（多 worker 并行时建议 False，避免 wandb 多 run 干扰）
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

N_WORKERS="${N_WORKERS:-2}"
GPU_IDS_STR="${GPU_IDS:-0 1}"
TOTAL_TRIALS="${TOTAL_TRIALS:-40}"
NUM_SEEDS="${NUM_SEEDS:-2}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
METRIC_LAST_N="${METRIC_LAST_N:-20}"
AGG="${AGG:-average}"
STORAGE="${STORAGE:-sqlite:///cleanrl_mujoco_unified_hpopt.db}"
STUDY_NAME="${STUDY_NAME:-mujoco_unified_$(date +%Y%m%d_%H%M%S)}"
TRACK_WANDB="${TRACK_WANDB:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-cleanrl_mujoco_unified_tune}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# parse GPU_IDS array
IFS_BACKUP="${IFS}"
IFS=' '
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
IFS="${IFS_BACKUP}"
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "GPU_IDS 不能为空（例如 GPU_IDS=\"0 1\"）"
  exit 1
fi

extra_args=(
  --num-seeds "${NUM_SEEDS}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --metric-last-n "${METRIC_LAST_N}"
  --aggregation "${AGG}"
  --storage "${STORAGE}"
  --study-name "${STUDY_NAME}"
)

if [[ "${TRACK_WANDB}" == "True" || "${TRACK_WANDB}" == "true" || "${TRACK_WANDB}" == "1" ]]; then
  extra_args+=(--track-wandb --wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    extra_args+=(--wandb-entity "${WANDB_ENTITY}")
  fi
fi

echo "========================================================"
echo "CleanRL unified MuJoCo tuning (multi-workers)"
echo "STUDY_NAME=${STUDY_NAME}"
echo "STORAGE=${STORAGE}"
echo "N_WORKERS=${N_WORKERS} | TOTAL_TRIALS=${TOTAL_TRIALS}"
echo "NUM_SEEDS=${NUM_SEEDS} | TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "GPU_IDS=${GPU_IDS[*]}"
echo "TRACK_WANDB=${TRACK_WANDB} | WANDB_PROJECT=${WANDB_PROJECT} | WANDB_ENTITY=${WANDB_ENTITY:-<none>}"
echo "========================================================"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

# 先初始化一次 Optuna storage/study，避免多 worker 并发建表的竞态条件导致 “table already exists”
echo "Initializing Optuna storage/study (to avoid sqlite race)..."
python scripts/tune_ppo_mujoco_unified.py \
  --num-trials 0 \
  --worker-id "init" \
  "${extra_args[@]}" \
  > "${LOG_DIR}/tune_mujoco_unified_${STUDY_NAME}_init.log" 2>&1 || true

pids=()
cleanup() {
  echo -e "\nCaught Ctrl+C / TERM, killing all workers..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
  echo "All workers killed."
  exit 0
}
trap cleanup INT TERM

for ((w=0; w< N_WORKERS; w++)); do
  gpu="${GPU_IDS[$(( w % ${#GPU_IDS[@]} ))]}"
  # 均分总 trials：base + remainder
  base=$(( TOTAL_TRIALS / N_WORKERS ))
  rem=$(( TOTAL_TRIALS % N_WORKERS ))
  trials_this_worker="${base}"
  if [[ "${w}" -lt "${rem}" ]]; then
    trials_this_worker=$(( base + 1 ))
  fi

  echo "Launching worker ${w}/${N_WORKERS} on GPU ${gpu} (trials=${trials_this_worker}) ..."
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/tune_ppo_mujoco_unified.py \
    --num-trials "${trials_this_worker}" \
    --worker-id "${w}" \
    "${extra_args[@]}" \
    > "${LOG_DIR}/tune_mujoco_unified_${STUDY_NAME}_w${w}.log" 2>&1 &
  pids+=($!)
done

echo "All workers started: ${pids[*]}"
wait
echo "All workers finished."

