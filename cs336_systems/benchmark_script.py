import torch
import timeit
import logging
import argparse
import numpy as np
from argparse import ArgumentParser, Namespace

from cs336_basics.model import BasicsTransformerLM

def setup_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument("--vocab_size", type=int, default=10000)
    args.add_argument("--context_length", type=int, default=512)
    args.add_argument("--d_model", type=int, default=768)
    args.add_argument("--num_layers", type=int, default=12)
    args.add_argument("--num_heads", type=int, default=12)
    args.add_argument("--d_ff", type=int, default=3072)
    args.add_argument("--w_steps", type=int, default=5, help="Number of warm-up steps before measuring time")
    args.add_argument("--n_steps", type=int, default=10, help="Number of measurement steps")
    args.add_argument("--device", type=torch.device, required=True, help="e.g., cpu, cuda:0, cuda:1, ...")
    args.add_argument("--sync", action=argparse.BooleanOptionalAction, required=True, help="Whether to call torch.cuda.synchronize() after each step")
    args.add_argument("--func_to_time", type=str, required=True, choices=["forward", "forward_backward", "forward_backward_optimizer"])
    return args.parse_args()

def forward_only(lm: BasicsTransformerLM, x: torch.Tensor, sync: bool = True, device: str = torch.device("cuda:0")) -> None:
    _ = lm(x)
    if sync:
        torch.cuda.synchronize(device)

func_to_time = {
    "forward": forward_only,
    "forward_backward": NotImplemented,
    "forward_backward_optimizer": NotImplemented
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # NOTE: hard-coded for now
    args = setup_args()
    print(f"Measuring performance on device: {args.device} (sync == {args.sync})")
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
    # Select the function to time
    func = func_to_time[args.func_to_time]
    # Warm-up steps
    for _ in range(args.w_steps):
        func(lm, x, sync=True if args.device != torch.device("cpu") else False, device=args.device)
    # Measurement
    per_step_secs = [None] * args.n_steps
    for i in range(args.n_steps):
        start = timeit.default_timer()
        func(lm, x, sync=args.sync, device=args.device)
        elapsed_secs = timeit.default_timer() - start
        per_step_secs[i] = elapsed_secs
    print(per_step_secs)
    print(f"Mean: {np.mean(per_step_secs):.4f} seconds; Std: {np.std(per_step_secs)} seconds")
