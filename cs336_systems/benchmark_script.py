import json
import uuid
import torch
import timeit
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from pprint import pprint
from functools import partial
from torch.optim.optimizer import Optimizer
from argparse import ArgumentParser, Namespace

from cs336_basics.model import BasicsTransformerLM

from pathlib import Path
HERE = Path(__file__).resolve().parent  # .../cs336_systems
REPO = HERE.parent

def setup_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument("--vocab_size", type=int, default=10000)
    args.add_argument("--context_length", type=int, default=512)
    args.add_argument("--d_model", type=int, default=768)
    args.add_argument("--num_layers", type=int, default=12)
    args.add_argument("--num_heads", type=int, default=12)
    args.add_argument("--d_ff", type=int, default=3072)
    args.add_argument("--model_size", type=str, default=None, help="Model size to benchmark. If provided, will override the hyperparameters specified above. Choices: small, medium, large, xl, 10B")
    args.add_argument("--w_steps", type=int, default=5, help="Number of warm-up steps before measuring time")
    args.add_argument("--n_steps", type=int, default=10, help="Number of measurement steps")
    args.add_argument("--device", type=torch.device, required=True, help="e.g., cpu, cuda:0, cuda:1, ...")
    args.add_argument("--sync", action=argparse.BooleanOptionalAction, required=True, help="Whether to call torch.cuda.synchronize() after each step")
    args.add_argument("--func_to_time", type=str, required=True, choices=["forward", "forward_backward", "forward_backward_optimizer"])
    args.add_argument("--debug", action="store_true", help="Debug mode.")
    args.add_argument("--sweep_name", type=str, required=True, help="Name of the sweep. Will be used to create the output directory.")
    return args.parse_args()

def update_args_with_model_size(args: Namespace, model_size: str) -> Namespace:
    df = pd.read_csv(HERE / "configs" / "model_sizes.csv", index_col=0)
    hparams = df.loc[model_size].to_dict()
    for k, v in hparams.items():
        setattr(args, k, v)
    return args

def forward(lm: BasicsTransformerLM, x: torch.Tensor) -> None:
    _ = lm(x)

def forward_backward(lm: BasicsTransformerLM, x: torch.Tensor, y: torch.Tensor) -> None:
    logits = lm(x)  # shape = (batch_size, seq_length, vocab_size)
    loss = torch.nn.functional.cross_entropy(input=logits.view(-1, logits.shape[-1]), target=y.view(-1))
    loss.backward()

def forward_backward_optimizer(lm: BasicsTransformerLM, x: torch.Tensor, y: torch.Tensor, optimizer: Optimizer) -> None:
    logits = lm(x)  # shape = (batch_size, seq_length, vocab_size)
    loss = torch.nn.functional.cross_entropy(input=logits.view(-1, logits.shape[-1]), target=y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def get_provenance_info() -> dict:
    import sys, shlex, socket
    from datetime import datetime, timezone
    def git(*cmd: str) -> str:
        return subprocess.check_output(["git", *cmd], cwd=REPO, text=True).strip()

    provenance = {
        "git_commit": git("rev-parse", "HEAD"),
        "command": shlex.join(sys.argv),   # import shlex, sys
        "hostname": socket.gethostname(),  # import socket
        "gpu": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return provenance

def build_record(args: Namespace, provenance: dict, per_step_secs: list[float], status: str) -> dict:
    return {
        **provenance,
        "model_size": args.model_size,
        "func_to_time": args.func_to_time,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "w_steps": args.w_steps,
        "n_steps": args.n_steps,
        "sync": args.sync,
        "device": str(args.device),
        "per_step_secs": [float(x) for x in per_step_secs],
        "mean_sec": np.mean(per_step_secs),
        "std_sec": np.std(per_step_secs),
        "status": status,
    }

def save_record(record: dict, folder_name: str) -> Path:
    out_dir = HERE / "benchmark_script_results" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.model_size}__{args.func_to_time}__{uuid.uuid4().hex[:8]}.json"  # NOTE: uuid is used to ensure unique file name
    out_path = out_dir / name
    with open(out_path, "w") as f:
        json.dump(record, f, indent=4)
    return out_path

def benchmark_function(args: Namespace) -> list[float]:
    lm = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(args.device)
    # Generate a random batch of data
    x = torch.randint(low=0, high=args.vocab_size, size=(4, 512)).to(args.device)
    y = torch.randint(low=0, high=args.vocab_size, size=(4, 512)).to(args.device)
    optimizer = torch.optim.AdamW(params=lm.parameters())
    # Select the function to time
    match args.func_to_time:
        case "forward":
            func = partial(forward, lm, x)
        case "forward_backward":
            func = partial(forward_backward, lm, x, y)
        case "forward_backward_optimizer":
            func = partial(forward_backward_optimizer, lm, x, y, optimizer)
    # Warm-up steps
    for _ in range(args.w_steps):
        func()
    torch.cuda.synchronize(args.device)
    # Measurement
    per_step_secs = [None] * args.n_steps
    for i in range(args.n_steps):
        start = timeit.default_timer()
        func()
        if args.sync:
            torch.cuda.synchronize(args.device)
        elapsed_secs = timeit.default_timer() - start
        per_step_secs[i] = elapsed_secs
    print(per_step_secs)
    print(f"Mean: {np.mean(per_step_secs):.4f} seconds; Std: {np.std(per_step_secs)} seconds")
    return [float(x) for x in per_step_secs]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # NOTE: hard-coded for now
    args = setup_args()
    not_commited = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    is_git_tree_dirty = bool(not_commited)
    if is_git_tree_dirty and not args.debug:
        print("Git tree is dirty. Please commit your changes before running the benchmark.")
        exit(1)  # NOTE: exit status 1 indicates failure
    if args.model_size is not None:
        print(f"Updating args with model size: {args.model_size}")
        args = update_args_with_model_size(args, args.model_size)
    print(f"Measuring performance of {args.func_to_time} on device: {args.device} (sync == {args.sync})")
    pprint(f"Args: {vars(args)}")
    try:
        per_step_secs = benchmark_function(args)
        status = "success"
    except Exception as e:
        print(f"Error: {e}")
        per_step_secs = []
        status = "error"
    if args.debug:
        status = "debug"
    provenance = get_provenance_info()
    record = build_record(args, provenance, per_step_secs, status)
    out_path = save_record(record, folder_name=args.sweep_name)
    print(f"Saved record to {out_path}")
