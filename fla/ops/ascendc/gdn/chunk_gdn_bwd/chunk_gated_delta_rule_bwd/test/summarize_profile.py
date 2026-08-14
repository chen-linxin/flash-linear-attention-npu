import argparse
import csv
from pathlib import Path


CORE_PREFIXES = (
    "RecomputeWUFwd_",
    "ChunkGatedDeltaRuleFwdH_",
    "ChunkBwdDvLocal_",
    "ChunkGatedDeltaRuleBwdDhu_",
    "ChunkBwdDqkwg_",
    "PrepareWyReprBwdDa_",
    "PrepareWyReprBwdFull_",
    "PrepareWyReprBwd_",
    "ChunkLocalCumsum_",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()

    rows: list[tuple[str, float]] = []
    for path in args.profile.glob("OPPROF_*/**/OpBasicInfo*.csv"):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                name = row["Op Name"]
                duration = float(row["Task Duration(us)"])
                if name.startswith(CORE_PREFIXES):
                    rows.append((name.split("_", 1)[0], duration))

    if not rows:
        raise RuntimeError(f"No core GDN kernels found under {args.profile}")

    total = sum(duration for _, duration in rows)
    for name, duration in rows:
        print(f"{name},{duration:.6f}")
    print(f"CORE_TOTAL,{total:.6f}")
    print(f"LAUNCHES,{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
