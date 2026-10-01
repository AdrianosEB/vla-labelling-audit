"""End-to-end audit on a synthetic corpus with a known amount of planted noise.

A sanity check, not an experiment: the corpus is built with a chosen fraction
of wrong labels, and the audit should recover roughly that fraction.

Run:  python scripts/demo_synthetic.py
"""

from __future__ import annotations

import numpy as np

from vla_label_audit import (
    bootstrap_alpha_ci,
    cca_alignment,
    cosine_distance_matrix,
    exact_match_distance_matrix,
    fit_degradation_curve,
    inject_label_noise,
    instruction_space_report,
    krippendorff_alpha,
    neighborhood_disagreement,
    neighborhood_overlap,
    predicted_cost,
    rank_correlation_across_views,
)

N_EPISODES, N_TASKS, DIM, LATENT = 600, 12, 48, 6
PLANTED_NOISE = 0.20


def build_corpus(seed: int = 0):
    """Episodes generated from a task latent, so vision and language truly align."""
    rng = np.random.default_rng(seed)
    task_latent = rng.normal(size=(N_TASKS, LATENT))
    task_of = rng.integers(0, N_TASKS, size=N_EPISODES)

    w_vis, w_lang = rng.normal(size=(LATENT, DIM)), rng.normal(size=(LATENT, DIM))
    z = task_latent[task_of]
    vision = z @ w_vis + 0.30 * rng.normal(size=(N_EPISODES, DIM))
    language = task_latent @ w_lang                      # one canonical phrasing per task
    return task_of, vision, language, w_lang, task_latent


def main() -> None:
    task_of, vision, task_lang, w_lang, task_latent = build_corpus()
    rng = np.random.default_rng(1)

    # Ground truth: each episode's label is its own task, then we corrupt some.
    noisy = inject_label_noise(task_of, PLANTED_NOISE, mode="shuffle", seed=7)
    truly_wrong = set(noisy.corrupted_idx.tolist())
    print(f"planted noise: {noisy.realised_rate:.1%} ({len(truly_wrong)} of {N_EPISODES} episodes)")

    # Annotations: 3 crowdworkers per episode, each paraphrasing the assigned label.
    label_emb = task_lang[noisy.labels]
    ann_unit, ann_emb, ann_text = [], [], []
    for i in range(N_EPISODES):
        for j in range(3):
            ann_unit.append(i)
            ann_emb.append(label_emb[i] + 0.25 * rng.normal(size=DIM))
            ann_text.append(f"ep{i}-worker{j}-phrasing")     # every string unique
    ann_unit, ann_emb = np.array(ann_unit), np.array(ann_emb)

    print("\n=== 1. do annotators agree? ===")
    d_sem = cosine_distance_matrix(ann_emb)
    sem = krippendorff_alpha(ann_unit, d_sem)
    nom = krippendorff_alpha(ann_unit, exact_match_distance_matrix(ann_text))
    point, lo, hi = bootstrap_alpha_ci(ann_unit, d_sem, n_boot=400, seed=0)
    print(f"  semantic alpha  {sem.alpha:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"  string alpha    {nom.alpha:.4f}")
    print("  -> the gap between them is paraphrase, not disagreement")

    print("\n=== 2. how much variety is in the language? ===")
    rep = instruction_space_report(ann_emb, ann_text)
    print(f"  unique strings         {rep['unique_fraction']:.1%}")
    print(f"  effective rank         {rep['effective_rank']:.1f}  (built from {N_TASKS} tasks)")
    print(f"  mean NN similarity     {rep['mean_nearest_neighbor_similarity']:.3f}")

    print("\n=== 3. do vision and language line up? ===")
    ep_lang = np.stack([ann_emb[ann_unit == i].mean(0) for i in range(N_EPISODES)])
    clean_lang = task_lang[task_of]
    for name, lang in (("as labelled", ep_lang), ("if labels were clean", clean_lang)):
        overlap = neighborhood_overlap(vision, lang, k=15).mean()
        rho = rank_correlation_across_views(vision, lang, sample=N_EPISODES)
        cca = cca_alignment(vision, lang, n_components=6)
        print(f"  {name:<22} overlap={overlap:.3f}  spearman={rho:+.3f}  {cca}")

    print("\n=== 4. can we find the wrong labels? ===")
    scores = neighborhood_disagreement(vision, ep_lang, k=15)
    order = np.argsort(-scores)
    for frac in (0.10, 0.20, 0.30):
        top = order[: int(frac * N_EPISODES)]
        hits = len(truly_wrong & set(top.tolist()))
        prec = hits / len(top)
        rec = hits / max(1, len(truly_wrong))
        print(f"  top {frac:.0%} most suspect: precision {prec:.1%}, recall {rec:.1%}")
    base = len(truly_wrong) / N_EPISODES
    top20 = order[: int(0.20 * N_EPISODES)]
    lift = (len(truly_wrong & set(top20.tolist())) / len(top20)) / base
    print(f"  random baseline precision would be {base:.1%} -> lift x{lift:.2f}")

    print("\n=== 5. what does the noise cost? (simulated policy) ===")
    rates = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    scores_sim = 0.72 - 0.55 * rates + rng.normal(0, 0.01, rates.size)
    curve = fit_degradation_curve(rates, scores_sim)
    print(f"  slope {curve['slope']:.3f} per unit noise (p={curve['p_value']:.2g})")
    print(
        f"  at the measured {noisy.realised_rate:.1%} noise rate that predicts a "
        f"{predicted_cost(curve, noisy.realised_rate) * 100:.1f} point cost"
    )
    print("\n  (step 5 is simulated here; on the real project it comes from training runs)")


if __name__ == "__main__":
    main()
