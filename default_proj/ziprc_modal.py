"""Modal launcher for the ZIP-RC-Lite pipeline.

Mirrors modal_train.py's image/volume/secret setup but targets ziprc/ scripts and
defaults to a CHEAP GPU (0.5B doesn't need H100). GPU only where it's needed:
  gen / train_head / score  -> GPU
  label / select            -> CPU-only (compute_score + Haiku API; no GPU spend)

Examples (run from default_proj/):
  modal run ziprc_modal.py gen   -- --out /vol/ziprc/data/rollouts.parquet --max-num-prompts 200 --samples-per-prompt 4
  modal run ziprc_modal.py label -- --in-parquet /vol/ziprc/data/rollouts.parquet --out-parquet /vol/ziprc/data/labeled.parquet --judge heuristic
  modal run ziprc_modal.py train -- --data-path /vol/ziprc/data/labeled.parquet --weights-path /vol/ziprc/models/lite_binary --label-column correct --reward-values 0.0 1.0 --max-steps 300
  modal run ziprc_modal.py score -- --model /vol/ziprc/models/lite_binary --in-parquet /vol/ziprc/data/labeled.parquet --out-parquet /vol/ziprc/data/scored.parquet --reward-values 0.0 1.0
  modal run ziprc_modal.py select -- --in-parquet /vol/ziprc/data/scored.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

import modal

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_ROOT = Path("/root/default_proj")
REMOTE_VOLUME_ROOT = Path("/vol")
REMOTE_REQUIREMENTS_PATH = REMOTE_PROJECT_ROOT / "modal_requirements.txt"

APP_NAME = os.environ.get("MODAL_APP_NAME", "ziprc-lite")
GPU_CONFIG = os.environ.get("ZIPRC_GPU", os.environ.get("MODAL_GPU", "A10G"))  # cheap default
TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", "3600"))  # short dev default
STARTUP_TIMEOUT_SECONDS = int(os.environ.get("MODAL_STARTUP_TIMEOUT_SECONDS", "1800"))
CPU_COUNT = int(os.environ.get("MODAL_CPU_COUNT", "8"))
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "default-proj-training")
PIP_EXTRA_INDEX_URL = os.environ.get("MODAL_PIP_EXTRA_INDEX_URL", "https://download.pytorch.org/whl/cu128")

TRAINING_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _secrets():
    vals = {}
    for k in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "ANTHROPIC_API_KEY", "ZIPRC_JUDGE_MODEL"):
        v = os.environ.get(k)
        if v:
            vals[k] = v
    return [modal.Secret.from_dict(vals)] if vals else []


# Exclude heavy/unneeded local dirs from the build context (the RLOO ckpt lives on
# the mounted volume, not in the image).
_IGNORE = [
    ".venv", ".venv/**", "downloaded_checkpoints", "downloaded_checkpoints/**",
    "logs", "logs/**", "figures", "figures/**", "__pycache__", "**/__pycache__/**",
    "*.pyc", "**/*.pyc", ".git", ".git/**", ".DS_Store", "**/.DS_Store", ".env",
]

# Layer order matters: install the heavy deps BEFORE copying code so that editing
# ziprc/ only re-runs the cheap code-copy + editable-install layers, not pip.
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("python -m pip install --upgrade pip==25.3 setuptools==80.10.2 wheel==0.46.3")
    .add_local_file(str(LOCAL_PROJECT_ROOT / "modal_requirements.txt"),
                    remote_path="/root/modal_requirements.txt", copy=True)
    .run_commands(
        f"python -m pip install --extra-index-url {shlex.quote(PIP_EXTRA_INDEX_URL)} "
        "-r /root/modal_requirements.txt"
    )
    .add_local_dir(str(LOCAL_PROJECT_ROOT), remote_path=str(REMOTE_PROJECT_ROOT),
                   copy=True, ignore=_IGNORE)
    .run_commands(f"cd {shlex.quote(str(REMOTE_PROJECT_ROOT))} && python -m pip install --no-deps -e .")
)

app = modal.App(APP_NAME)


def _run(script: str, script_args: list[str]) -> str:
    (REMOTE_VOLUME_ROOT / "ziprc" / "data").mkdir(parents=True, exist_ok=True)
    (REMOTE_VOLUME_ROOT / "ziprc" / "models").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    hf_home = REMOTE_VOLUME_ROOT / "cache" / "huggingface"
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    env.setdefault("PYTHONPATH", str(REMOTE_PROJECT_ROOT))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("WANDB__SERVICE_WAIT", "300")
    cmd = ["python", script, *script_args]
    print(f"[ziprc_modal] {shlex.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=str(REMOTE_PROJECT_ROOT), env=env, check=True)
    finally:
        TRAINING_VOLUME.commit()
    return f"finished {script}"


def _gpu_fn(fn):
    return app.function(image=base_image, gpu=GPU_CONFIG, cpu=CPU_COUNT, timeout=TIMEOUT_SECONDS,
                        startup_timeout=STARTUP_TIMEOUT_SECONDS,
                        volumes={str(REMOTE_VOLUME_ROOT): TRAINING_VOLUME}, secrets=_secrets())(fn)


def _cpu_fn(fn):
    return app.function(image=base_image, cpu=CPU_COUNT, timeout=TIMEOUT_SECONDS,
                        volumes={str(REMOTE_VOLUME_ROOT): TRAINING_VOLUME}, secrets=_secrets())(fn)


@_gpu_fn
def run_gen(a): return _run("ziprc/gen_rollouts.py", a)


@_cpu_fn
def run_label(a): return _run("ziprc/label_rollouts.py", a)


@_gpu_fn
def run_train(a): return _run("ziprc/train_head_only.py", a)


@_gpu_fn
def run_score(a): return _run("ziprc/score_joint_head.py", a)


@_cpu_fn
def run_select(a): return _run("ziprc/value_select.py", a)


@_gpu_fn
def run_decode(a): return _run("ziprc/adaptive_decode.py", a)


@_cpu_fn
def run_figures(a): return _run("ziprc/make_figures.py", a)


@_cpu_fn
def run_adaptive_k(a): return _run("ziprc/adaptive_k.py", a)


@_cpu_fn
def run_aggregate(a): return _run("ziprc/aggregate_pareto.py", a)


@_gpu_fn
def run_calib_tv(a): return _run("ziprc/calibration_tv.py", a)


# Long-running multi-stage pipeline that runs ENTIRELY in one remote container, so the
# sequence survives the local client disconnecting (e.g. laptop sleep). Launch with
# `modal run --detach`. Each step commits the volume, so partial progress is preserved.
PIPELINE_TIMEOUT = int(os.environ.get("ZIPRC_PIPELINE_TIMEOUT", "21600"))  # 6h


@app.function(image=base_image, gpu=GPU_CONFIG, cpu=CPU_COUNT, timeout=PIPELINE_TIMEOUT,
              startup_timeout=STARTUP_TIMEOUT_SECONDS,
              volumes={str(REMOTE_VOLUME_ROOT): TRAINING_VOLUME}, secrets=_secrets())
def run_pipeline(steps_json: str) -> str:
    # CONTINUE-ON-FAILURE: a failed step (e.g. a new/risky script) is logged but does NOT
    # abort the run, so validated high-value steps still complete unattended.
    steps = json.loads(steps_json)
    log = []
    for i, (script, sargs) in enumerate(steps):
        print(f"\n[pipeline] === step {i + 1}/{len(steps)}: {script} ===", flush=True)
        try:
            _run(script, sargs)
            log.append(f"OK   step {i + 1} {script}")
        except Exception as e:
            log.append(f"FAIL step {i + 1} {script}: {e}")
            print(f"[pipeline] step {i + 1} FAILED (continuing): {e}", flush=True)
    print("\n[pipeline] SUMMARY:\n" + "\n".join(log), flush=True)
    return "\n".join(log)


_POLICY = ("/vol/checkpoints/rloo_checkpoints/rloo_training/"
           "rloo_from_sft_gs16_bs64_lr1e5_clip1_iwclip_20260524_101017/latest_checkpoint/model")
_SFT = "asingh15/qwen-sft-countdown-defaultproj"
_DS = "asingh15/countdown_tasks_3to4"


def _build_pipeline(name: str):
    D, M = "/vol/ziprc/data", "/vol/ziprc/models"
    if name == "scaleup":
        return [
            ["ziprc/gen_rollouts.py", ["--model", _POLICY, "--dataset", _DS, "--split", "train",
                                       "--out", f"{D}/train_rollouts_512.parquet",
                                       "--max-num-prompts", "512", "--samples-per-prompt", "4"]],
            ["ziprc/label_rollouts.py", ["--in-parquet", f"{D}/train_rollouts_512.parquet",
                                         "--out-parquet", f"{D}/train_labeled_512.parquet", "--judge", "heuristic"]],
            ["ziprc/train_head_only.py", ["--model-id", _POLICY, "--data-path", f"{D}/train_labeled_512.parquet",
                                          "--weights-path", f"{M}/lite_binary_512", "--label-column", "correct",
                                          "--reward-values", "0.0", "1.0", "--batch-size", "16",
                                          "--gradient-accumulation-steps", "2", "--num-epochs", "3"]],
            ["ziprc/gen_rollouts.py", ["--model", _POLICY, "--dataset", _DS, "--split", "test",
                                       "--out", f"{D}/test_rollouts_256.parquet",
                                       "--max-num-prompts", "256", "--samples-per-prompt", "8"]],
            ["ziprc/label_rollouts.py", ["--in-parquet", f"{D}/test_rollouts_256.parquet",
                                         "--out-parquet", f"{D}/test_labeled_256.parquet", "--judge", "heuristic"]],
            ["ziprc/score_joint_head.py", ["--model", f"{M}/lite_binary_512", "--in-parquet", f"{D}/test_labeled_256.parquet",
                                           "--out-parquet", f"{D}/test_scored_256.parquet", "--reward-values", "0.0", "1.0"]],
            ["ziprc/value_select.py", ["--in-parquet", f"{D}/test_scored_256.parquet", "--ks", "1", "2", "4", "8", "16"]],
            ["ziprc/adaptive_decode.py", ["--model", f"{M}/lite_binary_512", "--dataset", _DS, "--split", "test",
                                          "--num-prompts", "80", "--K", "8", "--reward-values", "0.0", "1.0",
                                          "--warmup", "96", "--prune-interval", "32", "--keep-min", "1",
                                          "--betas", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1",
                                          "--out-parquet", f"{D}/pareto_256.parquet",
                                          "--pareto-out", f"{D}/pareto_summary_256.json"]],
            ["ziprc/make_figures.py", ["--scored", f"{D}/test_scored_256.parquet",
                                       "--pareto", f"{D}/pareto_summary_256.json", "--out-dir", "/vol/ziprc/figures_256"]],
        ]
    if name == "scaleup_plus":
        H = f"{M}/lite_binary_512"           # the scaled head (trained below)
        dec = lambda seed, out: ["ziprc/adaptive_decode.py", [
            "--model", H, "--dataset", _DS, "--split", "test", "--num-prompts", "40", "--K", "8",
            "--reward-values", "0.0", "1.0", "--warmup", "96", "--prune-interval", "32", "--keep-min", "1",
            "--betas", "0.02", "--stop-thresholds", "0.8", "--seed", str(seed), "--pareto-out", out]]
        return [
            # --- scaleup core (all validated code) ---
            ["ziprc/gen_rollouts.py", ["--model", _POLICY, "--dataset", _DS, "--split", "train",
                                       "--out", f"{D}/train_rollouts_512.parquet", "--max-num-prompts", "512", "--samples-per-prompt", "4"]],
            ["ziprc/label_rollouts.py", ["--in-parquet", f"{D}/train_rollouts_512.parquet", "--out-parquet", f"{D}/train_labeled_512.parquet", "--judge", "heuristic"]],
            ["ziprc/train_head_only.py", ["--model-id", _POLICY, "--data-path", f"{D}/train_labeled_512.parquet",
                                          "--weights-path", H, "--label-column", "correct", "--reward-values", "0.0", "1.0",
                                          "--batch-size", "16", "--gradient-accumulation-steps", "2", "--num-epochs", "3"]],
            ["ziprc/gen_rollouts.py", ["--model", _POLICY, "--dataset", _DS, "--split", "test",
                                       "--out", f"{D}/test_rollouts_256.parquet", "--max-num-prompts", "256", "--samples-per-prompt", "8"]],
            ["ziprc/label_rollouts.py", ["--in-parquet", f"{D}/test_rollouts_256.parquet", "--out-parquet", f"{D}/test_labeled_256.parquet", "--judge", "heuristic"]],
            ["ziprc/score_joint_head.py", ["--model", H, "--in-parquet", f"{D}/test_labeled_256.parquet", "--out-parquet", f"{D}/test_scored_256.parquet", "--reward-values", "0.0", "1.0"]],
            ["ziprc/value_select.py", ["--in-parquet", f"{D}/test_scored_256.parquet", "--ks", "1", "2", "4", "8", "16"]],
            # --- multi-seed Pareto (validated decode x3) + error bars ---
            dec(0, f"{D}/pareto_s0.json"), dec(1, f"{D}/pareto_s1.json"), dec(2, f"{D}/pareto_s2.json"),
            ["ziprc/aggregate_pareto.py", ["--inputs", f"{D}/pareto_s0.json", f"{D}/pareto_s1.json", f"{D}/pareto_s2.json",
                                           "--out-json", f"{D}/pareto_agg.json", "--out-png", "/vol/ziprc/figures_256/pareto_errorbars.png"]],
            # --- adaptive-K (validated) ---
            ["ziprc/adaptive_k.py", ["--scored", f"{D}/test_scored_256.parquet", "--budgets", "2", "3", "4", "5", "6",
                                     "--kmax", "8", "--trials", "16", "--out-json", f"{D}/adaptive_k_256.json"]],
            # --- exploratory: cross-policy transfer (validated code, mismatched head/data) ---
            ["ziprc/score_joint_head.py", ["--model", f"{M}/lite_binary_sft", "--in-parquet", f"{D}/test_labeled.parquet",
                                           "--out-parquet", f"{D}/cross_rloodata_sfthead.parquet", "--reward-values", "0.0", "1.0"]],
            ["ziprc/value_select.py", ["--in-parquet", f"{D}/cross_rloodata_sfthead.parquet", "--ks", "1", "4", "8"]],
            ["ziprc/score_joint_head.py", ["--model", H, "--in-parquet", f"{D}/sft_test_labeled.parquet",
                                           "--out-parquet", f"{D}/cross_sftdata_rloohead.parquet", "--reward-values", "0.0", "1.0"]],
            ["ziprc/value_select.py", ["--in-parquet", f"{D}/cross_sftdata_rloohead.parquet", "--ks", "1", "4", "8"]],
            # --- figures (validated + extended) ---
            ["ziprc/make_figures.py", ["--scored", f"{D}/test_scored_256.parquet", "--pareto", f"{D}/pareto_s0.json",
                                       "--adaptive-k", f"{D}/adaptive_k_256.json", "--out-dir", "/vol/ziprc/figures_256"]],
            # --- proposal deliverable: K=64 ground-truth TV calibration (new code, last) ---
            ["ziprc/gen_rollouts.py", ["--model", _POLICY, "--dataset", _DS, "--split", "test",
                                       "--out", f"{D}/k64_rollouts.parquet", "--max-num-prompts", "32", "--samples-per-prompt", "64"]],
            ["ziprc/label_rollouts.py", ["--in-parquet", f"{D}/k64_rollouts.parquet", "--out-parquet", f"{D}/k64_labeled.parquet", "--judge", "heuristic"]],
            ["ziprc/calibration_tv.py", ["--model", H, "--in-parquet", f"{D}/k64_labeled.parquet", "--reward-values", "0.0", "1.0",
                                         "--min-k", "32", "--out-json", f"{D}/calib_tv.json"]],
        ]
    raise ValueError(f"unknown pipeline: {name}")


@app.local_entrypoint()
def main(*raw):
    raw = list(raw)
    if raw and raw[0] == "pipeline":
        rest = raw[1:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        name = rest[0] if rest else "scaleup"
        steps = _build_pipeline(name)
        print(f"[pipeline] launching '{name}' ({len(steps)} steps) in one remote container")
        print(run_pipeline.remote(json.dumps(steps)))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("gen", "label", "train", "score", "select",
                                          "decode", "figures", "adaptive_k", "aggregate", "calib_tv"))
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    ns = parser.parse_args(raw)
    a = ns.rest[1:] if ns.rest[:1] == ["--"] else ns.rest
    fn = {"gen": run_gen, "label": run_label, "train": run_train, "score": run_score,
          "select": run_select, "decode": run_decode, "figures": run_figures,
          "adaptive_k": run_adaptive_k, "aggregate": run_aggregate, "calib_tv": run_calib_tv}[ns.stage]
    print(fn.remote(a))
