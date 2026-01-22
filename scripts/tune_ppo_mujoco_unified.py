"""
CleanRL PPO (continuous action) - multi-environment unified hyperparameter tuning.

目标：寻找“一套超参数”在多个 MuJoCo 任务上都表现不错（而不是每个任务单独 tune）。

实现方式：
- 用 Optuna 采样一组 PPO 超参
- 对每个 trial：对 5 个环境 * 多个 seed 运行 `cleanrl/ppo_continuous_action.py`
- 读取 TensorBoard 里的 `charts/episodic_return`，按各 env 的 target_scores 做归一化，再聚合成一个标量作为 objective

说明：
- CleanRL 官方文档也强调 Tuner 的用途是“多任务找一套超参”：
  docs/advanced/hyperparameter-tuning.md
- 我们在这里没有直接复用 cleanrl_utils.tuner.Tuner，
  主要原因是它用 `--flag=value` 形式拼参数，不方便传 `--no-xxx` 这类 bool flags。
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from tensorboard.backend.event_processing import event_accumulator


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


@dataclass(frozen=True)
class EnvTargetScore:
    low: float
    high: float

    def normalize(self, x: float) -> float:
        # Avoid divide-by-zero
        denom = (self.high - self.low) if (self.high - self.low) != 0 else 1.0
        return (x - self.low) / denom


def _read_tb_metric(run_name: str, metric: str, last_n: int) -> float:
    """
    Read last_n scalar values from tensorboard logs under runs/{run_name}, return their average.
    """
    ea = event_accumulator.EventAccumulator(f"runs/{run_name}")
    ea.Reload()
    scalars = ea.Scalars(metric)
    if len(scalars) == 0:
        raise RuntimeError(f"No scalars found for metric={metric} in runs/{run_name}")
    values = [s.value for s in scalars[-last_n:]] if last_n > 0 else [s.value for s in scalars]
    return float(np.average(values))


def _bool_flag(name: str, value: bool) -> List[str]:
    """
    Turn a boolean into tyro-compatible flags: --name or --no-name.
    """
    return [f"--{name}" if value else f"--no-{name}"]


def _format_trial_tag(trial: optuna.Trial, params: Dict[str, object]) -> str:
    # Keep it short-ish: trial number + a couple key params
    lr = params.get("learning-rate", None)
    steps = params.get("num-steps", None)
    envs = params.get("num-envs", None)
    return f"t{trial.number}_lr{lr}_ns{steps}_ne{envs}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--storage", type=str, default="sqlite:///cleanrl_mujoco_unified_hpopt.db")
    parser.add_argument("--study-name", type=str, default="")
    parser.add_argument(
        "--worker-id",
        type=str,
        default="",
        help="多 worker 并行时用于区分日志/（可选）wandb 的 worker 标识；不影响 Optuna 的 trial 分配",
    )
    parser.add_argument("--metric", type=str, default="charts/episodic_return")
    parser.add_argument("--metric-last-n", type=int, default=20)
    parser.add_argument("--aggregation", type=str, choices=["average", "median", "min"], default="average")
    parser.add_argument(
        "--track-wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否把每个 trial 的聚合分数记录到 W&B（需要本地已配置 wandb）",
    )
    parser.add_argument("--wandb-project", type=str, default="cleanrl_mujoco_unified_tune")
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    env_ids = [
        "HalfCheetah-v4",
        "Hopper-v4",
        "Walker2d-v4",
        "Swimmer-v4",
        "Humanoid-v4",
    ]

    # 这些 boundary 只是“量级级别”的 ballpark，用于跨环境归一化（不要求精确）。
    # 如果你有自己跑出来的 reward 尺度，建议按你的尺度改。
    target_scores: Dict[str, EnvTargetScore] = {
        "HalfCheetah-v4": EnvTargetScore(low=0.0, high=12_000.0),
        "Hopper-v4": EnvTargetScore(low=0.0, high=3_500.0),
        "Walker2d-v4": EnvTargetScore(low=0.0, high=6_000.0),
        "Swimmer-v4": EnvTargetScore(low=0.0, high=400.0),
        "Humanoid-v4": EnvTargetScore(low=0.0, high=6_000.0),
    }

    if args.aggregation == "average":
        agg_fn = np.average
    elif args.aggregation == "median":
        agg_fn = np.median
    else:
        agg_fn = np.min

    study_name = args.study_name or f"tune_mujoco_unified_{int(time.time())}"
    worker_suffix = f"_w{args.worker_id}" if args.worker_id else ""

    wandb_run = None
    if args.track_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{study_name}{worker_suffix}",
            group=study_name,
            config={
                "env_ids": env_ids,
                "target_scores": {k: asdict(v) for k, v in target_scores.items()},
                "metric": args.metric,
                "metric_last_n": args.metric_last_n,
                "total_timesteps": args.total_timesteps,
                "num_trials": args.num_trials,
                "num_seeds": args.num_seeds,
                "aggregation": args.aggregation,
            },
        )

    def objective(trial: optuna.Trial) -> float:
        # ---- search space (focus on lr + batch/rollout sizing + epochs) ----
        # 按你的要求：不调 gamma/gae_lambda/max_grad_norm/vf_coef/clip_coef（以及这里顺手不调 ent_coef），全部使用脚本默认值。
        params: Dict[str, object] = {
            "learning-rate": trial.suggest_float("learning-rate", 3e-5, 1e-3, log=True),
            "num-envs": trial.suggest_categorical("num-envs", [1, 4, 8]),
            "num-steps": trial.suggest_categorical("num-steps", [512, 1024, 2048, 4096]),
            "update-epochs": trial.suggest_categorical("update-epochs", [5, 10, 20]),
            "num-minibatches": trial.suggest_categorical("num-minibatches", [16, 32, 64]),
            # fixed for tuning speed / stability
            "total-timesteps": int(args.total_timesteps),
            "cuda": True,
            "track": False,  # 由本脚本聚合；不逐个 experiment 上 W&B，避免刷屏
            "capture-video": False,
            "save-model": False,
            "upload-model": False,
        }

        trial_tag = _format_trial_tag(trial, params)

        # Each seed produces one aggregated score (across envs), then average across seeds.
        per_seed_scores: List[float] = []

        for seed in range(args.num_seeds):
            per_env_norm_scores: List[float] = []
            for env_id in env_ids:
                # Build argv for tyro
                argv: List[str] = []
                # tyro 对 bool 参数更兼容 `--flag/--no-flag`，而不是 `--flag=True/False`
                for k, v in params.items():
                    if isinstance(v, bool):
                        argv += _bool_flag(k, v)
                    else:
                        argv += [f"--{k}={v}"]
                argv += [f"--env-id={env_id}", f"--seed={seed}", f"--exp-name={study_name}_{trial_tag}"]

                # Execute training script in-process
                sys.argv = argv
                with HiddenPrints():
                    experiment = runpy.run_path(path_name="cleanrl/ppo_continuous_action.py", run_name="__main__")

                run_name = experiment.get("run_name", None)
                if not run_name:
                    raise RuntimeError("Could not find run_name from executed script (cleanrl/ppo_continuous_action.py)")

                avg_return = _read_tb_metric(run_name, args.metric, args.metric_last_n)
                norm_score = target_scores[env_id].normalize(avg_return)
                per_env_norm_scores.append(norm_score)

            seed_score = float(agg_fn(per_env_norm_scores))
            per_seed_scores.append(seed_score)
            trial.report(seed_score, step=seed)
            if trial.should_prune():
                raise optuna.TrialPruned()

        final_score = float(np.average(per_seed_scores))
        if wandb_run:
            wandb_run.log(
                {
                    "trial": trial.number,
                    "trial_tag": trial_tag,
                    "score": final_score,
                    **{f"hp/{k}": v for k, v in params.items() if isinstance(v, (int, float, str, bool))},
                }
            )
        return final_score

    # 允许多 worker（多个进程）共享同一个 storage+study_name
    # 多进程首次并发初始化 sqlite 时，可能出现并发建表竞态（table already exists）。
    # 这里做一个轻量重试，避免 worker “秒退”。
    study = None
    for attempt in range(10):
        try:
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage=args.storage,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
                sampler=optuna.samplers.TPESampler(),
                load_if_exists=True,
            )
            break
        except Exception as e:
            msg = str(e)
            if "table studies already exists" in msg or "already exists" in msg:
                time.sleep(0.2 * (attempt + 1))
                continue
            raise
    assert study is not None
    print("==========================================================================================")
    print("CleanRL unified MuJoCo tuning")
    print(f"study_name={study_name}")
    if args.worker_id:
        print(f"worker_id={args.worker_id}")
    print(f"storage={args.storage}")
    print(f"envs={env_ids}")
    print(f"metric={args.metric} (last_n={args.metric_last_n}), aggregation={args.aggregation}")
    print("==========================================================================================")

    if args.num_trials <= 0:
        print("num_trials<=0: 初始化完成，未运行任何 trial。")
        return

    study.optimize(objective, n_trials=args.num_trials)
    print(f"Best value: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")

    if wandb_run:
        wandb_run.summary["best_value"] = study.best_trial.value
        for k, v in study.best_trial.params.items():
            wandb_run.summary[f"best/{k}"] = v
        wandb_run.finish(quiet=True)


if __name__ == "__main__":
    main()

