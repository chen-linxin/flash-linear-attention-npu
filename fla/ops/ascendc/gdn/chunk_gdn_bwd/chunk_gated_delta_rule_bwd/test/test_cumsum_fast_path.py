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


@dataclass(frozen=True)
class CumsumCase:
    name: str
    heads: int
    tokens: int
    chunk_size: int
    reverse: bool
    scale: float = 1.0
    cu_seqlens: tuple[int, ...] | None = None


CASES = (
    CumsumCase("F01_ONE_TOKEN", 1, 1, 64, False),
    CumsumCase("F02_TAIL_63", 2, 63, 64, False),
    CumsumCase("F03_TAIL_65_REVERSE", 3, 65, 64, True),
    CumsumCase("F04_CHUNK128_SCALE", 2, 129, 128, True, 0.5),
    CumsumCase("V01_SHORT_TAIL", 2, 64, 64, False, cu_seqlens=(0, 1, 64)),
    CumsumCase("V02_MULTI_CHUNK_REVERSE", 3, 129, 64, True, cu_seqlens=(0, 1, 65, 129)),
)


def cumsum_reference(x: torch.Tensor, case: CumsumCase) -> torch.Tensor:
    out = torch.empty_like(x)
    cu_seqlens = case.cu_seqlens or (0, case.tokens)
    for seq_start, seq_end in zip(cu_seqlens, cu_seqlens[1:]):
        for chunk_start in range(seq_start, seq_end, case.chunk_size):
            chunk_end = min(chunk_start + case.chunk_size, seq_end)
            chunk = x[..., chunk_start:chunk_end]
            if case.reverse:
                chunk = torch.flip(torch.cumsum(torch.flip(chunk, (-1,)), -1), (-1,))
            else:
                chunk = torch.cumsum(chunk, -1)
            out[..., chunk_start:chunk_end] = chunk * case.scale
    return out


def run_case(case: CumsumCase, device: str) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026)
    x_cpu = torch.randn(1, case.heads, case.tokens, generator=generator, dtype=torch.float32).mul_(0.1)
    x = x_cpu.to(device)
    cu_seqlens = None
    chunk_indices = None
    if case.cu_seqlens is not None:
        cu_seqlens = torch.tensor(case.cu_seqlens, device=device, dtype=torch.int64)
        metadata_g = x.transpose(1, 2).contiguous()
        cu_seqlens, _, chunk_indices, _ = gdr._ensure_varlen_metadata(
            metadata_g,
            cu_seqlens,
            list(case.cu_seqlens),
            None,
            None,
            case.chunk_size,
        )

    actual = gdr.chunk_local_cumsum_ascendc(
        x,
        chunk_size=case.chunk_size,
        reverse=case.reverse,
        scale=case.scale,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        output_dtype=torch.float32,
    ).cpu()
    expected = cumsum_reference(x_cpu, case)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    print(f"{case.name}: PASS max_abs={(actual - expected).abs().max().item():.9g}")


def expect_value_error(name: str, invoke) -> None:
    try:
        invoke()
    except ValueError:
        print(f"{name}: PASS")
    else:
        raise AssertionError(f"{name}: expected ValueError")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    for case in CASES:
        run_case(case, device)

    x = torch.zeros(1, 1, 2, device=device, dtype=torch.float32)
    expect_value_error("E01_RANK", lambda: gdr.chunk_local_cumsum_ascendc(x.squeeze(0), chunk_size=64))
    expect_value_error(
        "E02_HEAD_FIRST",
        lambda: gdr.chunk_local_cumsum_ascendc(x, chunk_size=64, head_first=False),
    )
    print(f"ALL {len(CASES)} CUMSUM CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
