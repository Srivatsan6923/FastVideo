"""
LoRA training MFU (Model FLOPs Utilization) calculation.

Reuses the same structure and metrics as mfu_calculation.py but runs
LoRA finetuning (--lora_training True, --lora_rank 32) and reports MFU.

Key difference from full-finetuning MFU: LoRA freezes base model weights,
which removes dL/dW for base projections/MLP but NOT dL/dx (activation
gradients still backpropagate through frozen layers into the adapters).
This creates an asymmetry in backward costs:
  - Frozen projections/MLP: backward = 1× forward  (dL/dx only)
  - Attention kernels:       backward = 2× forward  (inherent to softmax(QK^T)V)
whereas in full finetuning both are uniformly 2× forward.
"""

import os
import sys
from pathlib import Path

# Set Python path to current folder
current_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
os.environ["PYTHONPATH"] = current_dir + ":" + os.environ.get("PYTHONPATH", "")

import subprocess
import torch
import json
from huggingface_hub import snapshot_download
from fastvideo.utils import logger
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.training.wan_training_pipeline import WanTrainingPipeline

# Reused from mfu_calculation.py (same model/data)
MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_PATH = "data/crush-smol_processed_t2v/combined_parquet_dataset/worker_0"
VALIDATION_DATASET_FILE = "examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json"

# LoRA-specific output dir so we don't overwrite full finetune checkpoints
OUTPUT_DIR = Path("checkpoints/wan_t2v_finetune_lora_mfu")
PROFILER_TRACE_ROOT = Path("/workspace/profiler_traces")
MFU_SUMMARY_FILE = OUTPUT_DIR / "mfu_summary.json"
WANDB_SUMMARY_FILE = OUTPUT_DIR / "tracker/wandb/latest-run/files/wandb-summary.json"

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "1"
GRAD_ACCUM = "1"
LORA_RANK = "32"
MASTER_PORT = "29505"  # Different from vanilla MFU script so both can run in parallel

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = MASTER_PORT


def run_worker():
    """Worker function that will be run on each GPU (LoRA training)."""
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)

    # Same base args as mfu_calculation.py, plus LoRA
    args = parser.parse_args([
        "--model_path", MODEL_PATH,
        "--inference_mode", "False",
        "--pretrained_model_name_or_path", MODEL_PATH,
        "--data_path", DATA_PATH,
        "--dataloader_num_workers", "1",
        "--train_batch_size", "1",
        "--train_sp_batch_size", "1",
        "--gradient_accumulation_steps", GRAD_ACCUM,
        "--lora_training", "True",
        "--lora_rank", LORA_RANK,
        "--num_latent_t", "10",
        "--num_height", "480",
        "--num_width", "832",
        "--num_frames", "77",
        "--enable_gradient_checkpointing_type", "full",
        "--max_train_steps", "20",
        "--learning_rate", "5e-5",
        "--mixed_precision", "bf16",
        "--weight_only_checkpointing_steps", "250",
        "--training_state_checkpointing_steps", "250",
        "--weight_decay", "1e-4",
        "--max_grad_norm", "1.0",
        "--num_euler_timesteps", "50",
        "--multi_phased_distill_schedule", "4000-1",
        "--not_apply_cfg_solver",
        "--training_cfg_rate", "0.1",
        "--ema_start_step", "0",
        "--dit_precision", "fp32",
        "--output_dir", str(OUTPUT_DIR),
        "--tracker_project_name", "wan_t2v_finetune_lora",
        "--checkpoints_total_limit", "3",
        "--validation_dataset_file", VALIDATION_DATASET_FILE,
        "--validation_steps", "200",
        "--validation_sampling_steps", "50",
        "--validation_guidance_scale", "6.0",
        "--num_gpus", NUM_GPUS_PER_NODE,
        "--sp_size", NUM_GPUS_PER_NODE,
        "--tp_size", "1",
        "--hsdp_replicate_dim", NUM_GPUS_PER_NODE,
        "--hsdp_shard_dim", "1"
    ])
    pipeline = WanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("LoRA training pipeline done")


def _get_peak_flops_per_gpu(device_name: str) -> float:
    """Reused: peak BF16 TFLOPS per GPU."""
    if "H100" in device_name:
        return 989e12
    if "A100" in device_name:
        return 312e12
    if "A40" in device_name:
        return 312e12
    if "L40S" in device_name:
        return 362e12
    if "5090" in device_name:
        return 209.5e12
    raise ValueError(f"Device {device_name} not supported")


def _compute_mfu_from_summary(run_summary: dict, grad_accum: int,
                              lora_rank: int = 32) -> None:
    """Compute and print LoRA MFU from run summary."""
    device_name = torch.cuda.get_device_name()
    batch_size = run_summary.get("batch_size")
    seq_len = run_summary.get("dit_seq_len")
    context_len = run_summary.get("context_len")
    avg_step_time = run_summary.get("avg_step_time")
    hidden_dim = run_summary.get("hidden_dim")
    num_layers = run_summary.get("num_layers")
    ffn_dim = run_summary.get("ffn_dim")

    if any(v is None for v in (batch_size, seq_len, context_len, avg_step_time,
                               hidden_dim, num_layers, ffn_dim)):
        print("Could not calculate MFU: missing metrics in summary")
        return

    r = lora_rank

    # ---- Base model forward FLOPs per layer ----
    # Split into two categories because LoRA freezes base weights, which changes
    # the backward cost asymmetrically:
    #
    #   1. Projection / MLP FLOPs  (linear layers: y = x @ W)
    #      - Forward:  1 matmul  → cost F
    #      - Backward dL/dx (activation grad): 1 matmul → F   (same as fwd)
    #      - Backward dL/dW (weight grad):     SKIPPED (weights frozen in LoRA)
    #      ⇒ backward = 1× forward
    #
    #   2. Attention kernel FLOPs  (softmax(QK^T) @ V, no learnable params)
    #      - Forward:  2 matmuls (S=Q@K^T, O=A@V)                → cost F
    #      - Backward: 4 matmuls (dL/dV, dL/dA, dL/dQ, dL/dK)   → cost 2F
    #      ⇒ backward = 2× forward
    #      This 2× is inherent to the chain rule through softmax(QK^T)V: we need
    #      gradients w.r.t. Q, K, AND V, which requires 4 matmuls vs 2 in forward.
    #
    # In full finetuning both categories have backward = 2× forward (projections
    # add the dL/dW matmul), so you can lump everything together.  In LoRA the
    # frozen projections drop to 1× backward, creating the asymmetry.

    # Projection & MLP forward FLOPs (linear matmuls with learnable weights)
    qkv_out_flops = 8 * hidden_dim * hidden_dim * seq_len
    cross_attn_proj_flops = (
        (4 * hidden_dim * hidden_dim * seq_len) +
        (4 * hidden_dim * hidden_dim * context_len)
    )
    mlp_flops = 4 * hidden_dim * ffn_dim * seq_len
    proj_flops = qkv_out_flops + cross_attn_proj_flops + mlp_flops

    # Attention kernel forward FLOPs (no learnable parameters)
    self_attn_flops = 4 * seq_len * seq_len * hidden_dim
    cross_attn_flops = 4 * seq_len * context_len * hidden_dim
    attn_kernel_flops = self_attn_flops + cross_attn_flops

    base_fwd_per_layer = proj_flops + attn_kernel_flops

    # ---- Base model backward (activation gradients only, weights frozen) ----
    # Projections/MLP: dL/dx costs 1× forward (one transposed matmul per linear)
    # Attention kernel: dL/d{Q,K,V} costs 2× forward (4 matmuls vs 2 in fwd)
    base_bwd_act_per_layer = proj_flops + 2 * attn_kernel_flops

    # ---- LoRA forward addition FLOPs per layer ----
    # LoRA forward: delta = x @ A^T @ B^T  (two matmuls per LoRA'd linear)
    #   x @ A^T:  2 * tokens * in_dim * rank
    #   result @ B^T:  2 * tokens * rank * out_dim
    # For self-attn projections (q,k,v,o): in_dim = out_dim = hidden_dim, tokens = seq_len
    #   → 4 linears × 2 × seq_len × rank × (hidden_dim + hidden_dim)
    lora_self_attn_fwd = 4 * 2 * seq_len * r * (hidden_dim + hidden_dim)
    # For cross-attn projections (to_q, to_k, to_v, to_out):
    #   to_q, to_out: tokens=seq_len, in/out=hidden_dim
    #   to_k, to_v:   tokens=context_len, in/out=hidden_dim
    lora_cross_attn_fwd = (
        2 * 2 * seq_len * r * (hidden_dim + hidden_dim) +      # to_q, to_out
        2 * 2 * context_len * r * (hidden_dim + hidden_dim)     # to_k, to_v
    )
    lora_fwd_per_layer = lora_self_attn_fwd + lora_cross_attn_fwd

    # ---- LoRA backward FLOPs per layer ----
    # LoRA adapters ARE trainable, so backward includes both:
    #   dL/dx (activation grad):  same cost as forward matmul
    #   dL/dA (weight grad):      same cost as forward matmul
    # Two matmuls per LoRA'd linear → backward = 2× forward per LoRA'd linear
    lora_bwd_per_layer = 2 * lora_fwd_per_layer

    # ---- Total forward per layer (base + LoRA addition) ----
    fwd_per_layer = base_fwd_per_layer + lora_fwd_per_layer

    # ---- Total FLOPs per layer with activation checkpointing ----
    #
    # With gradient checkpointing the forward is run twice (once for the actual
    # forward pass, once recomputed during backward to reconstruct activations).
    #
    # Breakdown:
    #   1. Forward pass:          base_fwd + lora_fwd
    #   2. Recompute (ckpt):      base_fwd + lora_fwd
    #   3. Frozen base backward:  base_bwd_act  (dL/dx only, no dL/dW)
    #      - proj/MLP:    1× their forward  (one transposed matmul)
    #      - attn kernel:  2× their forward  (4 matmuls vs 2 in fwd)
    #   4. LoRA backward:         lora_bwd  (dL/dx + dL/dA + dL/dB)
    #
    # Simplified:
    #   = 2*(proj_mlp + attn_kernel + lora_fwd)     [fwd + recompute]
    #     + (proj_mlp + 2*attn_kernel)               [frozen base bwd]
    #     + 2*lora_fwd                               [LoRA bwd]
    #   = 3*proj_mlp + 4*attn_kernel + 4*lora_fwd
    total_per_layer = (
        2 * fwd_per_layer           # forward + recompute (activation checkpointing)
        + base_bwd_act_per_layer    # frozen base: activation grads only (no weight grads)
        + lora_bwd_per_layer        # LoRA: dL/dx + dL/dA + dL/dB
    )

    achieved_flops = batch_size * total_per_layer * num_layers
    achieved_flops *= grad_accum

    world_size = int(NUM_GPUS_PER_NODE)
    total_peak_flops = _get_peak_flops_per_gpu(device_name) * world_size
    achieved_flops_per_sec = achieved_flops / avg_step_time if avg_step_time > 0 else 0
    mfu = (achieved_flops_per_sec / total_peak_flops * 100) if total_peak_flops > 0 else 0

    lora_total_flops = 2 * lora_fwd_per_layer + lora_bwd_per_layer
    lora_frac = lora_total_flops / total_per_layer * 100
    print(f"LoRA Per-Step MFU: {mfu:.4f}%")
    print(f"  Proj+MLP fwd FLOPs/layer:   {proj_flops:.3e}")
    print(f"  Attn kernel fwd FLOPs/layer: {attn_kernel_flops:.3e}")
    print(f"  Base fwd FLOPs/layer:        {base_fwd_per_layer:.3e}")
    print(f"  Base bwd (act) FLOPs/layer:  {base_bwd_act_per_layer:.3e}  (proj/MLP 1x + attn 2x)")
    print(f"  LoRA fwd FLOPs/layer:        {lora_fwd_per_layer:.3e}  ({lora_fwd_per_layer/base_fwd_per_layer*100:.2f}% of base fwd)")
    print(f"  LoRA bwd FLOPs/layer:        {lora_bwd_per_layer:.3e}")
    print(f"  Total FLOPs/layer:           {total_per_layer:.3e}  (LoRA overhead: {lora_frac:.2f}% of total)")


def test_distributed_training(profile=False):
    """Run LoRA training in subprocess, then compute MFU."""
    os.environ["WANDB_MODE"] = "disabled"
    data_dir = Path("data/crush-smol_processed_t2v")
    if not data_dir.exists():
        print(f"Downloading test dataset to {data_dir}...")
        snapshot_download(
            repo_id="wlsaidhi/crush-smol_processed_t2v",
            local_dir=str(data_dir),
            repo_type="dataset",
            local_dir_use_symlinks=False
        )

    current_file = Path(__file__).resolve()

    cmd = []
    if profile:
        import datetime
        PROFILER_TRACE_ROOT.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_filename = f"{PROFILER_TRACE_ROOT}/trace_{timestamp}"
        cmd.extend([
            "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys", "profile",
            "--trace=cuda,nvtx",
            f"--output={trace_filename}",
            "--force-overwrite=true",
        ])

    cmd.extend([
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        "--master_port", MASTER_PORT,
        str(current_file)
    ])
    process = subprocess.run(cmd, capture_output=True, text=True)

    if process.stdout:
        print("STDOUT:", process.stdout)
    if process.stderr:
        print("STDERR:", process.stderr)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, cmd, process.stdout, process.stderr
        )

    # Reuse same summary resolution: mfu_summary.json (from --trackers none) or wandb
    summary_file = MFU_SUMMARY_FILE
    if not summary_file.exists():
        summary_file = WANDB_SUMMARY_FILE
    if not summary_file.exists():
        wandb_dir = OUTPUT_DIR / "tracker" / "wandb"
        candidates = list(wandb_dir.glob("**/wandb-summary.json"))
        if candidates:
            summary_file = max(candidates, key=lambda p: p.stat().st_mtime)
        else:
            raise FileNotFoundError(
                f"No summary found under {OUTPUT_DIR}. "
                "Expected mfu_summary.json or tracker/wandb/.../wandb-summary.json"
            )

    with summary_file.open() as f:
        run_summary = json.load(f)

    try:
        _compute_mfu_from_summary(run_summary, grad_accum=int(GRAD_ACCUM),
                                  lora_rank=int(LORA_RANK))
    except Exception as e:
        print(f"Could not calculate MFU: {e}")


if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is not None:
        run_worker()
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Run LoRA MFU calculation test")
        parser.add_argument("--profile", action="store_true",
                            help="Enable Nsight profiling with CUDA and NVTX tracing")
        args = parser.parse_args()
        test_distributed_training(profile=args.profile)
