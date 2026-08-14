import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu


REPO_ROOT = Path(__file__).resolve().parents[7]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import flash_gated_delta_rule as gdr  # noqa: E402
from fla_npu.ops.ascendc import (  # noqa: E402
    prepare_wy_repr_bwd_da,
    prepare_wy_repr_bwd_full,
)


@dataclass(frozen=True)
class PerfCase:
    name: str
    batch: int
    tokens: int
    query_heads: int
    value_heads: int
    value_dim: int
    chunk_size: int
    seq_num: int | None


CASES = {
    "P1": PerfCase("P1", 1, 65536, 32, 32, 128, 64, 64),
    "P2": PerfCase("P2", 1, 65536, 16, 32, 128, 64, 64),
    "P3": PerfCase("P3", 1, 65536, 16, 32, 256, 64, 64),
    "P4": PerfCase("P4", 8, 8192, 32, 32, 128, 128, None),
}


def balanced_cu_seqlens(total_tokens: int, seq_num: int) -> list[int]:
    base, remainder = divmod(total_tokens, seq_num)
    offsets = [0]
    for index in range(seq_num):
        offsets.append(offsets[-1] + base + (1 if index < remainder else 0))
    return offsets


def make_inputs(case: PerfCase, device: str):
    torch.manual_seed(2026)
    torch.npu.manual_seed_all(2026)
    dtype = torch.bfloat16
    key_dim = 128
    q = torch.randn(case.batch, case.query_heads, case.tokens, key_dim, device=device, dtype=dtype).mul_(0.01)
    k = torch.randn_like(q).mul_(0.01)
    v = torch.randn(
        case.batch, case.value_heads, case.tokens, case.value_dim, device=device, dtype=dtype
    ).mul_(0.01)
    # The backward entry consumes the chunk-local cumulative gate produced by
    # forward. Use a deterministic decay so long performance cases stay finite.
    chunk_positions = torch.arange(case.tokens, device=device, dtype=torch.float32)
    chunk_positions = chunk_positions.remainder(case.chunk_size).add_(1).mul_(-0.02)
    g = chunk_positions.view(1, case.tokens, 1).expand(
        case.batch, case.tokens, case.value_heads
    ).contiguous()
    beta = torch.full((case.batch, case.tokens, case.value_heads), 0.5, device=device, dtype=dtype)
    A = torch.zeros(
        case.batch,
        case.value_heads,
        case.tokens,
        case.chunk_size,
        device=device,
        dtype=dtype,
    )
    do = torch.randn(
        case.batch, case.tokens, case.value_heads, case.value_dim, device=device, dtype=dtype
    ).mul_(0.01)
    if case.seq_num is None:
        return q, k, v, g, beta, A, do, None, None, None, None

    cu_list = balanced_cu_seqlens(case.tokens, case.seq_num)
    cu_seqlens = torch.tensor(cu_list, device=device, dtype=torch.int64)
    cu_seqlens, cu_list, chunk_indices, chunk_lists = gdr._ensure_varlen_metadata(
        g,
        cu_seqlens,
        cu_list,
        None,
        None,
        case.chunk_size,
    )
    return q, k, v, g, beta, A, do, cu_seqlens, cu_list, chunk_indices, chunk_lists


def baseline_prepare_wy_repr_bwd(
    k,
    v,
    beta,
    A,
    dw,
    dv,
    g,
    chunk_size,
    cu_seqlens=None,
    chunk_indices=None,
):
    dA = prepare_wy_repr_bwd_da(
        k,
        v,
        beta.float(),
        A,
        dw,
        dv,
        g.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--implementation", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if args.implementation == "baseline":
        gdr.ascendc_prepare_wy_repr_bwd = baseline_prepare_wy_repr_bwd

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    device = f"npu:{args.device}"
    case = CASES[args.case]
    inputs = make_inputs(case, device)
    q, k, v, g, beta, A, do, cu_seqlens, cu_list, chunk_indices, chunk_lists = inputs

    def run_once():
        outputs = gdr.flash_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=128**-0.5,
            initial_state=None,
            do=do,
            dht=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_lists,
            chunk_size=case.chunk_size,
        )
        torch.npu.synchronize()
        return outputs

    with torch.no_grad():
        for _ in range(args.warmup):
            run_once()
        print(f"PROFILE_BEGIN case={case.name}", flush=True)
        for _ in range(args.repeat):
            outputs = run_once()
        print(f"PROFILE_END case={case.name}", flush=True)

    for output in outputs[:-1]:
        if not torch.isfinite(output.float()).all().item():
            raise AssertionError("profile output contains NaN or Inf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
