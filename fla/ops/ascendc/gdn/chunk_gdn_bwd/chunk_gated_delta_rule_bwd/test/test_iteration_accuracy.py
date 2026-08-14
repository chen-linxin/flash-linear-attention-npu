import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu


REPO_ROOT = Path(__file__).resolve().parents[7]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import flash_gated_delta_rule as gdr  # noqa: E402


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    batch: int
    query_heads: int
    value_heads: int
    tokens: int
    value_dim: int
    chunk_size: int
    dtype: str
    gate_function: str = "logsigmoid"
    qk_l2norm: bool = False
    cu_seqlens: tuple[int, ...] | None = None


CASES = (
    AccuracyCase("D01_ONE_TOKEN", 1, 1, 1, 1, 128, 64, "fp16"),
    AccuracyCase("D02_TAIL_63", 1, 1, 1, 63, 128, 64, "bf16"),
    AccuracyCase("D03_EXACT_64", 1, 2, 2, 64, 128, 64, "fp16"),
    AccuracyCase("D04_TAIL_65", 1, 2, 2, 65, 128, 64, "bf16"),
    AccuracyCase("D05_TAIL_127", 1, 2, 2, 127, 128, 64, "fp16"),
    AccuracyCase("D06_TWO_CHUNKS", 1, 2, 2, 128, 128, 64, "bf16"),
    AccuracyCase("D07_CHUNK128_TAIL", 1, 2, 2, 129, 128, 128, "fp16"),
    AccuracyCase("D08_BATCH2", 2, 2, 2, 64, 128, 64, "bf16"),
    AccuracyCase("D09_BATCH2_V256", 2, 2, 2, 128, 256, 128, "fp16"),
    AccuracyCase("D10_GVA_1_TO_2", 1, 1, 2, 64, 128, 64, "bf16"),
    AccuracyCase("D11_GVA_1_TO_3", 1, 1, 3, 65, 128, 64, "fp16"),
    AccuracyCase("D12_GVA_1_TO_4", 1, 1, 4, 128, 128, 64, "bf16"),
    AccuracyCase("D13_GVA_1_TO_5", 1, 1, 5, 127, 256, 128, "fp16"),
    AccuracyCase("D14_GVA_2_TO_6", 1, 2, 6, 129, 256, 128, "bf16"),
    AccuracyCase("D15_GVA_2_TO_16", 1, 2, 16, 64, 128, 64, "fp16"),
    AccuracyCase("D16_QK_L2NORM", 1, 2, 4, 64, 128, 64, "bf16", qk_l2norm=True),
    AccuracyCase("D17_NEGATIVE_LINEAR", 1, 2, 4, 64, 128, 64, "fp16", "negative_linear"),
    AccuracyCase("D18_ZERO_GATE", 1, 1, 1, 2, 128, 64, "bf16", "zeros"),
    AccuracyCase("D19_V256_CHUNK64", 1, 2, 2, 128, 256, 64, "fp16"),
    AccuracyCase("D20_V256_CHUNK128", 1, 2, 4, 256, 256, 128, "bf16"),
    AccuracyCase("V01_1_63", 1, 1, 1, 64, 128, 64, "fp16", cu_seqlens=(0, 1, 64)),
    AccuracyCase("V02_64_64", 1, 2, 2, 128, 128, 64, "bf16", cu_seqlens=(0, 64, 128)),
    AccuracyCase("V03_63_64_65", 1, 2, 4, 192, 128, 64, "fp16", cu_seqlens=(0, 63, 127, 192)),
    AccuracyCase("V04_1_65_127", 1, 2, 4, 193, 256, 128, "bf16", cu_seqlens=(0, 1, 66, 193)),
    AccuracyCase("V05_MANY_SHORT", 1, 1, 4, 64, 128, 64, "fp16", cu_seqlens=(0, 1, 2, 3, 64)),
    AccuracyCase("V06_127_1", 1, 2, 2, 128, 128, 128, "bf16", cu_seqlens=(0, 127, 128)),
    AccuracyCase("V07_129_63", 1, 2, 4, 192, 256, 128, "fp16", cu_seqlens=(0, 129, 192)),
    AccuracyCase("V08_GVA_1_TO_3", 1, 1, 3, 128, 128, 64, "bf16", cu_seqlens=(0, 31, 64, 128)),
    AccuracyCase("V09_GVA_V256", 1, 2, 6, 64, 256, 64, "fp16", cu_seqlens=(0, 31, 64)),
    AccuracyCase(
        "V10_GVA_L2NORM", 1, 1, 5, 128, 256, 128, "bf16", qk_l2norm=True, cu_seqlens=(0, 1, 65, 128)
    ),
)


def run_accuracy_case(case: AccuracyCase, device: str, cache_dir: str, force_regenerate: bool) -> None:
    cu_seqlens = None
    cu_text = ""
    if case.cu_seqlens is not None:
        cu_text = ",".join(str(value) for value in case.cu_seqlens)
        cu_seqlens = torch.tensor(case.cu_seqlens, device=device, dtype=torch.int64)

    args = SimpleNamespace(
        demo_model=False,
        initial_state="none",
        output_final_state=False,
        accuracy_tensors="o,dq,dk,dv,dbeta,dg",
        batch=case.batch,
        tokens=case.tokens,
        chunk_size=case.chunk_size,
        dtype=case.dtype,
        seed=2026,
        gate_function=case.gate_function,
        qk_l2norm=case.qk_l2norm,
        varlen=case.cu_seqlens is not None,
        cu_seqlens=cu_text,
        mean_len=64,
        accuracy_cache_dir=cache_dir,
        case_name=case.name,
        accuracy_force_regenerate=force_regenerate,
        accuracy_output_tol=5e-3,
        accuracy_grad_tol=8e-3,
        accuracy_beta_grad_tol=2e-2,
        accuracy_gate_grad_tol=2e-2,
        accuracy_output_cos_min=0.999,
        accuracy_grad_cos_min=0.999,
        accuracy_beta_grad_cos_min=0.99,
        accuracy_gate_grad_cos_min=0.99,
    )
    dtype = torch.float16 if case.dtype == "fp16" else torch.bfloat16
    gdr._run_accuracy_check(
        args,
        dtype=dtype,
        query_heads=case.query_heads,
        value_heads=case.value_heads,
        key_dim=128,
        value_dim=case.value_dim,
        scale=128**-0.5,
        device=device,
        cu_seqlens=cu_seqlens,
    )


def expect_value_error(name: str, invoke) -> None:
    try:
        invoke()
    except ValueError:
        print(f"{name}: PASS")
    else:
        raise AssertionError(f"{name}: expected ValueError")


def run_validation_cases() -> None:
    q = torch.zeros(1, 1, 2, 128, dtype=torch.float16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros(1, 2, 1, dtype=torch.float32)
    beta = torch.zeros(1, 2, 1, dtype=torch.float16)

    def call(q_=q, k_=k, v_=v, g_=g, beta_=beta, chunk_size=64):
        return gdr.flash_gated_delta_rule(q_, k_, v_, g_, beta_, chunk_size=chunk_size)

    expect_value_error("E01_MAIN_FP32", lambda: call(q.float(), k.float(), v.float()))
    expect_value_error("E02_DTYPE_MISMATCH", lambda: call(k_=k.bfloat16()))
    expect_value_error("E03_BETA_RANK", lambda: call(beta_=beta.squeeze(-1)))
    expect_value_error("E04_GATE_RANK", lambda: call(g_=g.squeeze(-1)))
    expect_value_error("E05_Q_RANK", lambda: call(q_=q.squeeze(0)))
    expect_value_error("E06_QK_PREFIX", lambda: call(k_=k.expand(1, 2, 2, 128)))
    expect_value_error("E07_QV_PREFIX", lambda: call(v_=v.expand(1, 2, 2, 128)))
    expect_value_error("E08_GATE_BETA_SHAPE", lambda: call(g_=g.expand(1, 2, 2)))
    expect_value_error("E09_GATE_HEAD", lambda: call(g_=torch.zeros(1, 2, 2), beta_=torch.zeros(1, 2, 2)))
    expect_value_error("E10_CHUNK_NOT_POWER_OF_TWO", lambda: call(chunk_size=96))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cases", default="")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    device = f"npu:{args.device}"
    wanted = {name.strip() for name in args.cases.split(",") if name.strip()}
    selected_cases = [case for case in CASES if not wanted or case.name in wanted]
    if not selected_cases:
        raise ValueError("No accuracy cases selected")
    failures: list[str] = []
    for index, case in enumerate(selected_cases, start=1):
        print(f"[{index}/{len(selected_cases)}] {case.name}", flush=True)
        try:
            run_accuracy_case(case, device, args.cache_dir, args.force_regenerate)
        except Exception as exc:
            failures.append(f"{case.name}: {exc}")
            print(f"{case.name}: FAIL: {exc}", flush=True)
            if args.stop_on_fail:
                break
        else:
            print(f"{case.name}: PASS", flush=True)

    if not failures and not args.skip_validation:
        run_validation_cases()
    if failures:
        print("FAILED CASES:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"ALL {len(selected_cases)} ACCURACY CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
