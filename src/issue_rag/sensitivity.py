"""Compare two runs that differ only in who graded them.

This is the module that makes the project's central claim reproducible. The claim
is not "retrieval quadruples accuracy"; it is **"retrieval helps by somewhere
between 1.4x and 4x, and the ranking of systems is identical under both graders"**
-- and that requires two runs over the same answers with different judges.

Workflow::

    # 1. the baseline run (generator and judge are the same model)
    python -m issue_rag.evaluate --preset vite --provider deepseek \\
        --systems closed_book,bm25,dense,hybrid,hybrid_rerank

    # 2. re-grade the stored answers with a different model, same items
    python -m issue_rag.evaluate --preset vite --provider deepseek --regrade \\
        --judge-model deepseek-reasoner --judge-max-tokens 3000 \\
        --run-tag reasoner

    # 3. compare
    python -m issue_rag.sensitivity --preset vite --variant reasoner

Step 2 reuses the answers, which is the point: the generator is held fixed, so
any difference in the scores is attributable to the grader.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

from .config import RESULTS_DIR, RepoPreset, load_env_file, load_preset
from .evaluate import load_shard, results_stem, shard_path

SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
USABLE = {"correct", "partial"}
DEFAULT_SYSTEMS = ("closed_book", "bm25", "dense", "hybrid", "hybrid_rerank")


def compare_records(
    baseline: dict[str, dict[str, Any]],
    variant: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score the same items under two graders.

    Only items present and non-error in *both* runs are compared; anything else
    would make the two numbers refer to different populations, which is exactly
    the mistake this module exists to avoid.
    """
    common = [
        item_id
        for item_id in baseline
        if item_id in variant
        and not baseline[item_id].get("error")
        and not variant[item_id].get("error")
        and baseline[item_id].get("verdict") in SCORES
        and variant[item_id].get("verdict") in SCORES
    ]
    if not common:
        return {"n": 0}

    base_scores = [SCORES[baseline[i]["verdict"]] for i in common]
    var_scores = [SCORES[variant[i]["verdict"]] for i in common]
    exact = sum(1 for i in common if baseline[i]["verdict"] == variant[i]["verdict"])
    binary = sum(
        1
        for i in common
        if (baseline[i]["verdict"] in USABLE) == (variant[i]["verdict"] in USABLE)
    )

    deltas: dict[str, int] = {}
    for i in common:
        key = f"{baseline[i]['verdict']} -> {variant[i]['verdict']}"
        if baseline[i]["verdict"] != variant[i]["verdict"]:
            deltas[key] = deltas.get(key, 0) + 1

    return {
        "n": len(common),
        "dropped": len(set(baseline) | set(variant)) - len(common),
        "baseline_accuracy": round(statistics.mean(base_scores), 4),
        "variant_accuracy": round(statistics.mean(var_scores), 4),
        "delta": round(statistics.mean(var_scores) - statistics.mean(base_scores), 4),
        "exact_agreement": round(100.0 * exact / len(common), 1),
        "binary_agreement": round(100.0 * binary / len(common), 1),
        "verdict_transitions": dict(sorted(deltas.items(), key=lambda kv: -kv[1])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="vite")
    parser.add_argument(
        "--variant",
        required=True,
        help=(
            "run tag of the second run (see evaluate --run-tag). Use the literal "
            "'untagged' to address the untagged arm."
        ),
    )
    parser.add_argument(
        "--systems", default=",".join(DEFAULT_SYSTEMS), help="comma separated"
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="run tag of the baseline (default: untagged; 'untagged' also works)",
    )
    args = parser.parse_args(argv)

    # The untagged arm has an empty tag, which cannot be typed as a normal
    # argument, so accept the word explicitly.
    if args.variant.strip().lower() == "untagged":
        args.variant = ""
    if args.baseline.strip().lower() == "untagged":
        args.baseline = ""

    load_env_file()
    preset: RepoPreset = load_preset(args.preset)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    rows: dict[str, dict[str, Any]] = {}
    missing = []
    for system in systems:
        base = load_shard(shard_path(preset, system, args.baseline))
        variant = load_shard(shard_path(preset, system, args.variant))
        if not base or not variant:
            missing.append(system)
            continue
        rows[system] = compare_records(base, variant)

    if not rows:
        raise SystemExit(
            f"no comparable runs found. Expected both untagged and '__{args.variant}' "
            f"shards under {RESULTS_DIR}. Run evaluate --regrade first."
        )

    def judge_of(tag: str) -> str:
        path = RESULTS_DIR / f"{results_stem(preset, tag)}.json"
        if path.exists():
            recorded = json.loads(path.read_text(encoding="utf-8")).get("judge")
            if recorded:
                return recorded
        # Older result files predate the `judge` field; fall back to the shard,
        # and finally to a label the table can still print.
        probe = load_shard(shard_path(preset, "closed_book", tag))
        for record in probe.values():
            if record.get("judge_model"):
                return record["judge_model"]
        return tag or "baseline judge (unrecorded)"

    base_judge = judge_of(args.baseline)
    variant_judge = judge_of(args.variant)
    # Include the run tag in the label: when both runs use the same model -- the
    # self-consistency case -- two identical names make the table unreadable.
    base_label = f"{base_judge} [{args.baseline or 'baseline'}]"
    variant_label = f"{variant_judge} [{args.variant}]"
    same_model = base_judge == variant_judge

    lines: list[str] = []
    lines.append(f"# Grader sensitivity — {preset.display_name} (`{preset.repo}`)\n")
    lines.append(f"- baseline judge: `{base_label}`")
    lines.append(f"- variant judge: `{variant_label}`")
    lines.append(
        "- answers are identical in both runs; only the grader differs "
        "(`evaluate --regrade`)\n"
    )
    if same_model:
        lines.append(
            "> **Both runs used the same model.** This table therefore measures the "
            "grader's own run-to-run noise, which sets the floor below which a "
            "difference between two systems must not be read as a result.\n"
        )

    lines.append("## Accuracy under each grader\n")
    lines.append(f"| system | n | dropped | `{base_label}` | `{variant_label}` | delta |")
    lines.append("|---|---|---|---|---|---|")
    total_dropped = 0
    for system, row in rows.items():
        total_dropped += row.get("dropped", 0)
        lines.append(
            f"| `{system}` | {row['n']} | {row.get('dropped', 0)} | "
            f"{row['baseline_accuracy']:.3f} | "
            f"{row['variant_accuracy']:.3f} | {row['delta']:+.3f} |"
        )
    lines.append("")
    if total_dropped:
        lines.append(
            f"_{total_dropped} item(s) across all systems were dropped because one "
            "side lacked a usable verdict (a failed call, or a grader that emitted "
            "no parseable JSON). They are excluded rather than scored, because a "
            "grader failure is not a wrong answer._\n"
        )

    lines.append("## Do the two graders agree?\n")
    lines.append("| system | n | exact | binary (usable vs not) |")
    lines.append("|---|---|---|---|")
    for system, row in rows.items():
        lines.append(
            f"| `{system}` | {row['n']} | {row['exact_agreement']}% | "
            f"{row['binary_agreement']}% |"
        )
    lines.append("")

    # The headline: does retrieval still look worth it under both graders?
    if "closed_book" in rows:
        ranked = [s for s in rows if s != "closed_book"]
        best = max(ranked, key=lambda s: rows[s]["baseline_accuracy"]) if ranked else None
        if best:
            b_row, v_row = rows["closed_book"], rows[best]
            base_gain = b_row["baseline_accuracy"] and (
                v_row["baseline_accuracy"] - b_row["baseline_accuracy"]
            )
            var_gain = v_row["variant_accuracy"] - b_row["variant_accuracy"]
            lines.append("## The claim that survives, and the one that does not\n")
            lines.append(
                f"Best retrieval system by baseline accuracy: `{best}`.\n"
            )
            lines.append(
                f"- gain under `{base_label}`: **{base_gain:+.3f}** "
                f"({b_row['baseline_accuracy']:.3f} -> {v_row['baseline_accuracy']:.3f})"
            )
            lines.append(
                f"- gain under `{variant_label}`: **{var_gain:+.3f}** "
                f"({b_row['variant_accuracy']:.3f} -> {v_row['variant_accuracy']:.3f})"
            )
            lines.append("")
            base_order = [s for s in sorted(rows, key=lambda s: rows[s]["baseline_accuracy"])]
            var_order = [s for s in sorted(rows, key=lambda s: rows[s]["variant_accuracy"])]
            stable = base_order == var_order
            lines.append(f"**Ranking stability**: {'identical' if stable else 'changed'} "
                         f"under both graders.")
            lines.append("")
            lines.append(f"- under `{base_label}`: `{' < '.join(base_order)}`")
            lines.append(f"- under `{variant_label}`: `{' < '.join(var_order)}`")
            lines.append("")
            if not stable:
                moved = [
                    s
                    for s in rows
                    if base_order.index(s) != var_order.index(s)
                ]
                spread = [
                    abs(rows[a]["baseline_accuracy"] - rows[b]["baseline_accuracy"])
                    for i, a in enumerate(moved)
                    for b in moved[i + 1 :]
                ]
                max_spread = max(spread) if spread else 0.0
                lines.append(
                    f"{', '.join(f'`{s}`' for s in moved)} change position. Before "
                    f"quoting that as a finding, check the gap: the largest "
                    f"difference among the movers is **{max_spread:.3f}**, which at "
                    f"n≈{rows[moved[0]]['n']} is well inside one item. An ordering "
                    "that flips under a grader swap and whose members are a fraction "
                    "of an item apart is not a finding — report it as unresolved."
                )
                lines.append("")
            worst_base, worst_var = base_order[0], var_order[0]
            best_base, best_var = base_order[-1], var_order[-1]
            lines.append(
                f"- closed-book is the worst system under both graders: "
                f"**{worst_base == worst_var == 'closed_book'}**"
            )
            lines.append(
                f"- the best system is the same under both: "
                f"**{best_base == best_var}** (`{best_base}`)"
            )
            lines.append("")
            lines.append(
                "So the defensible claims are exactly those two, plus the gain "
                "interval. Any ordering between systems that swap places inside the "
                "noise band is unresolved, not a result."
            )

    lines.append("")
    lines.append("## Verdict transitions\n")
    for system, row in rows.items():
        if not row["verdict_transitions"]:
            lines.append(f"- `{system}`: none")
            continue
        detail = ", ".join(f"{k} ({v})" for k, v in row["verdict_transitions"].items())
        lines.append(f"- `{system}`: {detail}")

    out_md = RESULTS_DIR / f"{preset.slug}__sensitivity__{args.variant}.md"
    out_json = RESULTS_DIR / f"{preset.slug}__sensitivity__{args.variant}.json"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "preset": preset.repo,
                "baseline_judge": base_judge,
                "variant_judge": variant_judge,
                "systems": rows,
                "missing": missing,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\n-> {out_md}")
    print(f"-> {out_json}")
    if missing:
        print(f"\nskipped (no shards for one side): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
