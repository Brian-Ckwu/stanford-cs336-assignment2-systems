import torch
import timeit
import logging
import argparse
import numpy as np
from functools import partial
from torch.optim.optimizer import Optimizer
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

def forward(lm: BasicsTransformerLM, x: torch.Tensor, sync: bool = True, device: str = torch.device("cuda:0")) -> None:
    _ = lm(x)
    if sync:
        torch.cuda.synchronize(device)

def forward_backward(lm: BasicsTransformerLM, x: torch.Tensor, y: torch.Tensor, sync: bool = True, device: str = torch.device("cuda:0")) -> None:
    logits = lm(x)  # shape = (batch_size, seq_length, vocab_size)
    loss = torch.nn.functional.cross_entropy(input=logits.view(-1, logits.shape[-1]), target=y.view(-1))
    loss.backward()
    if sync:
        torch.cuda.synchronize(device)

def forward_backward_optimizer(lm: BasicsTransformerLM, x: torch.Tensor, y: torch.Tensor, optimizer: Optimizer, sync: bool = True, device: str = torch.device("cuda:0")) -> None:
    logits = lm(x)  # shape = (batch_size, seq_length, vocab_size)
    loss = torch.nn.functional.cross_entropy(input=logits.view(-1, logits.shape[-1]), target=y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if sync:
        torch.cuda.synchronize(device)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # NOTE: hard-coded for now
    args = setup_args()
    print(f"Measuring performance of {args.func_to_time} on device: {args.device} (sync == {args.sync})")
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
            func = partial(forward, lm, x, sync=args.sync, device=args.device)
        case "forward_backward":
            func = partial(forward_backward, lm, x, y, sync=args.sync, device=args.device)
        case "forward_backward_optimizer":
            func = partial(forward_backward_optimizer, lm, x, y, optimizer, sync=args.sync, device=args.device)
    # Warm-up steps
    for _ in range(args.w_steps):
        func()
    # Measurement
    per_step_secs = [None] * args.n_steps
    for i in range(args.n_steps):
        start = timeit.default_timer()
        func()
        elapsed_secs = timeit.default_timer() - start
        per_step_secs[i] = elapsed_secs
    print(per_step_secs)
    print(f"Mean: {np.mean(per_step_secs):.4f} seconds; Std: {np.std(per_step_secs)} seconds")
