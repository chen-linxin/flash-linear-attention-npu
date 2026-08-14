import argparse
import sys
from pathlib import Path

import torch
import torch_npu


REPO_ROOT = Path(__file__).resolve().parents[7]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import flash_gated_delta_rule as gdr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--query-heads", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--value-dim", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(2026)
    torch.npu.manual_seed_all(2026)
    device = f"npu:{args.device}"
    dtype = torch.float16
    batch = 1
    key_dim = 128

    q = torch.randn(batch, args.query_heads, args.tokens, key_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(batch, args.value_heads, args.tokens, args.value_dim, device=device, dtype=dtype)
    g = torch.randn(batch, args.tokens, args.value_heads, device=device, dtype=torch.float32).clamp_max_(0)
    beta = torch.rand(batch, args.tokens, args.value_heads, device=device, dtype=dtype)
    A = torch.randn(
        batch,
        args.value_heads,
        args.tokens,
        args.chunk_size,
        device=device,
        dtype=dtype,
    ).mul_(0.01)
    do = torch.randn(batch, args.tokens, args.value_heads, args.value_dim, device=device, dtype=dtype)

    if args.full:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        g.requires_grad_(True)
        beta.requires_grad_(True)
        o, final_state = gdr.ChunkGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            key_dim**-0.5,
            None,
            False,
            None,
            None,
            None,
            None,
            False,
            args.chunk_size,
        )
        (o.float() * do.float()).sum().backward()
        torch.npu.synchronize()
        full_outputs = (o, q.grad, k.grad, v.grad, beta.grad, g.grad)
        for name, output in zip(("o", "dq", "dk", "dv", "dbeta", "dg"), full_outputs, strict=True):
            if output is None or not torch.isfinite(output.float()).all().item():
                raise AssertionError(f"{name} is missing or contains NaN/Inf")
            print(f"{name}: shape={tuple(output.shape)} dtype={output.dtype}")
        if final_state is not None:
            raise AssertionError("final_state must be None in the stateless mode")
        return 0

    outputs = gdr.flash_chunk_gated_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        scale=key_dim**-0.5,
        initial_state=None,
        do=do,
        dht=None,
        chunk_size=args.chunk_size,
    )
    torch.npu.synchronize()
    names = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
    for name, output in zip(names, outputs, strict=True):
        if output is None:
            print(f"{name}=None")
            continue
        if not torch.isfinite(output.float()).all().item():
            raise AssertionError(f"{name} contains NaN or Inf")
        print(f"{name}: shape={tuple(output.shape)} dtype={output.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
