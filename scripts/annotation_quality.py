"""Count defective annotations in DROID's annotation file.

Four defect classes:

* truncated: cut off mid-word ("Turn on the kett", "Close the cof").
* no terminal punctuation: reported for completeness. At ~91% it is the house
  style and not really a defect.
* non-answer: the annotator declined ("N/A", "Unsure", "No action").
* junk: non-linguistic ("+++++++", a bare "g").

The script also checks whether truncation comes from a character limit in the
annotation tool. A limit would show up as a spike in the length histogram and
as truncated annotations sharing one length. It tests both, plus association
with lab, collection date, and the length of the other annotations on the same
episode. The truncation detector uses no episode-level signal, so the
sibling-length test is not circular.

Semantic alpha is then recomputed with each class excluded in turn.

Run:
    python scripts/annotation_quality.py              # full file
    python scripts/annotation_quality.py --limit 5000 # quick pass
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from droid_agreement import CACHE, corpus_tag, embed, fetch_annotations, normalize  # noqa: E402

from vla_label_audit.scalable import alpha_semantic, bootstrap_alpha_semantic  # noqa: E402

WORD = re.compile(r"[a-z]+")
TERMINAL = frozenset(".!?")
SYSTEM_DICT = Path("/usr/share/dict/words")

# Whole-string non-answers, matched against lower-cased text. Anchored, so an
# instruction that only contains "none" does not match.
NON_ANSWER = re.compile(
    r"^(?:"
    r"n/?a|null|none|nan|nil|"
    r"unsure|not sure|unclear|unknown|idk|i don'?t know|can'?t tell|cannot tell|"
    r"no action|not action|no motion|no movement|no task|nothing|nothing happens|"
    r"no instruction|invalid|blank|empty"
    r")[.!?]?$"
)


def load_rows(raw: dict, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Every annotation slot that holds text, normalised but not filtered.

    Absent slots are skipped (12,500 episodes carry only `instruction1`),
    since a slot nobody was asked to fill is not a defective annotation.
    """
    episodes, slots, texts = [], [], []
    for i, (episode, fields) in enumerate(raw.items()):
        if limit and i >= limit:
            break
        for slot in (1, 2, 3):
            value = fields.get(f"language_instruction{slot}")
            if value is None:
                continue
            text = normalize(value)
            if not text:
                continue
            episodes.append(episode)
            slots.append(slot)
            texts.append(text)
    return np.array(episodes), np.array(slots), texts


def load_dictionary() -> set[str]:
    """The system word list, used only to veto truncation calls.

    /usr/share/dict/words is too sparse to detect truncation by itself (it
    lacks "box", "countertop", "laptop"), so it never marks a word as broken.
    It only clears words: if "stop" or "diagonal" is listed, it is not treated
    as a cut-off "stopper" or "diagonally".

    `flag_truncated` applies the veto only to tokens of four or more characters.
    The list includes every single letter and many obscure short words ("ta",
    "po", "gar", "pac"), so vetoing short tokens would discard the clearest
    truncations in the file ("in the clear b", "the silver bot").
    """
    if not SYSTEM_DICT.exists():
        return set()
    with SYSTEM_DICT.open(encoding="utf-8", errors="ignore") as fh:
        return {line.strip().lower() for line in fh if line.strip()}


def flag_truncated(texts: list[str], words: set[str]) -> np.ndarray:
    """Flag mid-word truncation using the corpus's own vocabulary.

    A truncated tail like "kett" is rare as a whole token but is a prefix of
    common tokens ("kettle"). The vocabulary is built from DROID itself, which
    is about 2,500 words of tabletop manipulation, so the test stays in-domain.

    An annotation is flagged only if it does not end in terminal punctuation,
    its last word appears at most twice as a whole word, and that word (if four
    or more characters) is not in the dictionary.

    Some calls stay ambiguous. "Move the mic" is flagged because "mic" occurs
    once as a whole word against ~1,000 longer "mic-" words, but it could be a
    complete instruction about a microphone. The flagged set is small, so the
    script prints all of it for inspection.
    """
    vocab: collections.Counter = collections.Counter()
    for text in texts:
        vocab.update(WORD.findall(text.lower()))

    extensions: collections.Counter = collections.Counter()
    for word, count in vocab.items():
        for cut in range(1, len(word)):
            extensions[word[:cut]] += count

    flags = np.zeros(len(texts), dtype=bool)
    for i, text in enumerate(texts):
        if text[-1] in TERMINAL:
            continue
        tail = WORD.findall(text.lower())
        if not tail:
            continue
        tail = tail[-1]
        if len(tail) >= 4 and tail in words:
            continue
        if vocab[tail] <= 2 and extensions[tail] >= 20:
            flags[i] = True
    return flags


def flag_no_terminal_punct(texts: list[str]) -> np.ndarray:
    return np.array([t[-1] not in TERMINAL for t in texts], dtype=bool)


def flag_non_answer(texts: list[str]) -> np.ndarray:
    return np.array([bool(NON_ANSWER.match(t.lower())) for t in texts], dtype=bool)


def flag_junk(texts: list[str]) -> np.ndarray:
    """Non-linguistic strings: no word content, or a single repeated character.

    Kept separate from non-answers: a non-answer means the annotator could not
    describe the episode, while junk is a bad input.
    """
    flags = np.zeros(len(texts), dtype=bool)
    for i, text in enumerate(texts):
        letters = re.sub(r"[^A-Za-z]", "", text)
        if len(letters) < 2:
            flags[i] = True
        elif len(set(text)) == 1:
            flags[i] = True
    return flags


def describe_lengths(texts: list[str]) -> None:
    chars = np.array([len(t) for t in texts])
    words = np.array([len(t.split()) for t in texts])
    print(f"  {'':14}{'min':>6}{'p05':>7}{'p25':>7}{'p50':>7}{'p75':>7}{'p95':>7}{'p99':>7}{'max':>7}")
    for name, arr in (("characters", chars), ("words", words)):
        q = np.percentile(arr, [5, 25, 50, 75, 95, 99]).astype(int)
        print(
            f"  {name:<14}{arr.min():>6}{q[0]:>7}{q[1]:>7}{q[2]:>7}{q[3]:>7}"
            f"{q[4]:>7}{q[5]:>7}{arr.max():>7}"
        )
    print(f"\n  mean {chars.mean():.1f} chars, {words.mean():.1f} words")

    # A hard tool limit would pile annotations up at the cap, so print the most
    # common lengths in the top decile.
    counts = collections.Counter(chars.tolist())
    tail = [(L, c) for L, c in counts.items() if L >= np.percentile(chars, 90)]
    tail.sort(key=lambda kv: -kv[1])
    dense = ", ".join(f"{L}ch x{c}" for L, c in tail[:8])
    print(f"  densest lengths in the top decile: {dense}")


def truncation_diagnostics(
    episodes: np.ndarray, texts: list[str], trunc: np.ndarray
) -> dict:
    """Test truncation against length, lab, date and sibling annotation length."""
    out: dict = {}
    n_trunc = int(trunc.sum())
    chars = np.array([len(t) for t in texts])

    print(f"\n  {n_trunc} truncated annotations of {len(texts):,} ({n_trunc/len(texts):.4%})")
    if n_trunc == 0:
        return out

    lengths = sorted(chars[trunc].tolist())
    print(f"  their lengths (chars): {lengths}")
    print(f"  distinct lengths: {len(set(lengths))} of {n_trunc}")
    out["truncated_lengths"] = lengths
    # A fixed-width cut would leave most truncated strings the same length.
    out["length_concentration"] = len(set(lengths)) / n_trunc

    print("\n  the truncated annotations:")
    for ep, text in sorted(zip(episodes[trunc].tolist(), [texts[i] for i in np.flatnonzero(trunc)])):
        print(f"     {ep.split('+')[0]:<9} {text!r}")

    # By lab.
    labs = np.array([e.split("+")[0] for e in episodes])
    table, names = [], []
    for lab in np.unique(labs):
        m = labs == lab
        table.append([int((m & trunc).sum()), int((m & ~trunc).sum())])
        names.append(lab)
    table = np.array(table)
    print("\n  by lab:")
    for name, (bad, ok) in zip(names, table):
        rate = bad / (bad + ok)
        print(f"     {name:<10} {bad:>3} / {bad+ok:>7,}  ({rate:.4%})")
    chi2, p_lab, _, expected = stats.chi2_contingency(table)
    low = int((expected < 5).sum())
    print(f"     chi2 = {chi2:.2f}, p = {p_lab:.3f}")
    if low:
        print(f"     WARNING: {low} of {expected.size} expected counts < 5; chi2 is unreliable here")
    out["lab_chi2_p"] = float(p_lab)
    out["lab_low_expected_cells"] = low

    # By collection month.
    months = np.array([e.split("+")[2][:7] for e in episodes])
    uniq = sorted(set(months.tolist()))
    mtable = np.array([[int(((months == m) & trunc).sum()),
                        int(((months == m) & ~trunc).sum())] for m in uniq])
    keep = mtable.sum(axis=1) >= 200
    chi2_m, p_month, _, exp_m = stats.chi2_contingency(mtable[keep])
    print(f"\n  by month ({int(keep.sum())} months with >=200 annotations):")
    for m, (bad, ok) in zip([u for u, k in zip(uniq, keep) if k], mtable[keep]):
        if bad:
            print(f"     {m}  {bad:>3} / {bad+ok:>6,}")
    print(f"     chi2 = {chi2_m:.2f}, p = {p_month:.3f}")
    low_m = int((exp_m < 5).sum())
    if low_m:
        print(f"     WARNING: {low_m} of {exp_m.size} expected counts < 5; chi2 is unreliable here")
    out["month_chi2_p"] = float(p_month)
    out["month_low_expected_cells"] = low_m

    # Sibling length. If a tool clipped the input, the other annotations on the
    # same episode should be longer than usual.
    by_ep: dict[str, list[int]] = collections.defaultdict(list)
    for i, ep in enumerate(episodes.tolist()):
        by_ep[ep].append(i)
    sibling = np.full(len(texts), np.nan)
    for idxs in by_ep.values():
        if len(idxs) < 2:
            continue
        lens = chars[idxs].astype(float)
        total = lens.sum()
        for j, i in enumerate(idxs):
            sibling[i] = (total - lens[j]) / (len(idxs) - 1)
    ok = ~np.isnan(sibling)
    a, b = sibling[ok & trunc], sibling[ok & ~trunc]
    if a.size and b.size:
        u, p_sib = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(
            f"\n  mean sibling length: {a.mean():.1f} chars for truncated (n={a.size}) "
            f"vs {b.mean():.1f} for the rest (n={b.size:,})"
        )
        print(f"     Mann-Whitney U p = {p_sib:.3f}")
        out["sibling_length_p"] = float(p_sib)
        out["sibling_mean_truncated"] = float(a.mean())
        out["sibling_mean_rest"] = float(b.mean())
    return out


def alpha_deltas(
    episodes: np.ndarray, emb: np.ndarray, classes: dict[str, np.ndarray], n_boot: int
) -> dict:
    """Semantic alpha with each defect class removed, one at a time."""
    base = alpha_semantic(episodes, emb)
    point, lo, hi = bootstrap_alpha_semantic(episodes, emb, n_boot=n_boot, seed=0)
    print(f"  baseline           alpha {base.alpha:+.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
          f"   ({base.n_units:,} episodes, {base.n_pairable:,} annotations)")

    out = {"baseline": {"alpha": base.alpha, "ci": [lo, hi],
                        "n_units": int(base.n_units), "n_annotations": int(base.n_pairable)}}
    for name, flags in classes.items():
        n_drop = int(flags.sum())
        if n_drop == 0:
            print(f"  excl. {name:<13} (none found, alpha unchanged)")
            out[name] = {"n_dropped": 0, "delta": 0.0}
            continue
        keep = ~flags
        try:
            res = alpha_semantic(episodes[keep], emb[keep])
            _, klo, khi = bootstrap_alpha_semantic(
                episodes[keep], emb[keep], n_boot=n_boot, seed=0
            )
        except ValueError as exc:
            print(f"  excl. {name:<13} not computable: {exc}")
            continue
        delta = res.alpha - base.alpha
        lost_units = base.n_units - res.n_units
        print(
            f"  excl. {name:<13} alpha {res.alpha:+.4f}  95% CI [{klo:.4f}, {khi:.4f}]"
            f"   delta {delta:+.4f}   (-{n_drop:,} annotations, -{lost_units:,} episodes)"
        )
        # Dropping most of the corpus gives a different corpus, so this delta is
        # not comparable to the others.
        if n_drop > 0.5 * flags.size:
            print(
                f"       ^ this drops {n_drop/flags.size:.0%} of all annotations and leaves "
                f"{res.n_units:,} episodes; not a defect class, and not comparable above"
            )
        out[name] = {
            "n_dropped": n_drop,
            "alpha": res.alpha,
            "ci": [klo, khi],
            "delta": delta,
            "episodes_lost": int(lost_units),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only use the first N episodes")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap replicates per alpha")
    args = ap.parse_args()

    raw = fetch_annotations()
    print(f"\nloaded {len(raw):,} episodes from the annotation file")
    episodes, slots, texts = load_rows(raw, args.limit)
    print(f"  {len(texts):,} non-empty annotations across {len(set(episodes.tolist())):,} episodes")

    print("\n" + "=" * 70)
    print("LENGTH DISTRIBUTION")
    print("=" * 70)
    describe_lengths(texts)

    words = load_dictionary()
    if not words:
        print("\n  note: /usr/share/dict/words missing; truncation veto disabled")
    trunc = flag_truncated(texts, words)
    noterm = flag_no_terminal_punct(texts)
    nonans = flag_non_answer(texts)
    junk = flag_junk(texts)

    print("\n" + "=" * 70)
    print("DEFECT CLASSES")
    print("=" * 70)
    for name, flags in (("truncated mid-word", trunc), ("no terminal punctuation", noterm),
                        ("explicit non-answer", nonans), ("non-linguistic junk", junk)):
        n = int(flags.sum())
        print(f"  {name:<26} {n:>7,}  ({n/len(texts):.4%})")

    if nonans.any():
        print("\n  non-answers:")
        for text, count in collections.Counter(
            texts[i] for i in np.flatnonzero(nonans)
        ).most_common():
            print(f"     {count:>3}x {text!r}")
        # An episode where every annotator wrote "No action" counts as perfect
        # agreement and inflates alpha, so report how many of those there are.
        grouped: dict[str, list[int]] = collections.defaultdict(list)
        for i, ep in enumerate(episodes.tolist()):
            grouped[ep].append(i)
        pairable = [idxs for idxs in grouped.values() if len(idxs) >= 2]
        touched = sum(1 for idxs in pairable if any(nonans[i] for i in idxs))
        whole = sum(1 for idxs in pairable if all(nonans[i] for i in idxs))
        print(
            f"     concentrated on {touched} multiply-annotated episodes, "
            f"{whole} of which are entirely non-answer"
        )
    if junk.any():
        print("\n  junk:")
        for text, count in collections.Counter(
            texts[i] for i in np.flatnonzero(junk)
        ).most_common(20):
            print(f"     {count:>3}x {text[:60]!r}")

    print("\n" + "=" * 70)
    print("IS TRUNCATION AN ANNOTATION-TOOL LIMIT?")
    print("=" * 70)
    diag = truncation_diagnostics(episodes, texts, trunc)

    print("\n" + "=" * 70)
    print("SEMANTIC ALPHA WITH EACH CLASS EXCLUDED")
    print("=" * 70)
    emb = embed(texts, tag=corpus_tag(texts))
    deltas = alpha_deltas(
        episodes,
        emb,
        {"truncated": trunc, "non-answer": nonans, "junk": junk, "no-term-punct": noterm},
        args.boot,
    )

    out = CACHE / "annotation_quality.json"
    out.write_text(
        json.dumps(
            {
                "n_annotations": len(texts),
                "n_episodes": len(set(episodes.tolist())),
                "counts": {
                    "truncated": int(trunc.sum()),
                    "no_terminal_punctuation": int(noterm.sum()),
                    "non_answer": int(nonans.sum()),
                    "junk": int(junk.sum()),
                },
                "truncation_diagnostics": diag,
                "alpha_excluding": deltas,
            },
            indent=2,
        )
    )
    print(f"\n\nresults written to {out}")


if __name__ == "__main__":
    main()
