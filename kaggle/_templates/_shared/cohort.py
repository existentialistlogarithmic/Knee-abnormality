"""Which studies exist, what their targets are, and how the folds are cut.

Included verbatim into generated kernels by eda/generate_kernels.py.
Kaggle script kernels are single files, so sharing code means splicing it
at generation time. Editing it here changes every kernel that includes it.

This is the most correctness-critical block in the project and the reason it is
shared rather than copied. An out-of-fold evaluation is only out-of-fold if the
evaluating kernel cuts the folds **exactly** as the training kernel did — same
study ordering, same grouping key, same GroupKFold call. A second copy that
sorted differently would produce a number that looks like a held-out score and
is not one, and nothing anywhere would raise.
"""


def build_cohort(cache_dirs, artifacts, headers_dir, competition, findings,
                 gold_weight=8.0, abstain_masks_loss=True, min_studies=50,
                 quiet=False):
    """Every study with a cached volume, a target and a scanner fingerprint.

    Returns a dict rather than a tuple: callers want different subsets of this
    and positional unpacking across two kernels is one more thing to get out of
    step.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold

    def say(*args):
        if not quiet:
            print(*args, flush=True)

    soft = pd.read_parquet(artifacts / "soft_labels.parquet").set_index("StudyInstanceUID")

    # study -> file, across every shard. A duplicate would mean the shards
    # overlap, which would silently over-weight those studies.
    cache_paths = {}
    duplicates = 0
    for directory in cache_dirs:
        for npy in directory.glob("*.npy"):
            if npy.stem in cache_paths:
                duplicates += 1
            cache_paths[npy.stem] = npy
    say(f"cached studies: {len(cache_paths):,}"
        + (f"   WARNING: {duplicates} duplicate studies across shards" if duplicates else ""))
    available = set(cache_paths)

    # The fold-grouping key. No site column exists anywhere in this dataset, so
    # folds group on a scanner fingerprint; random K-fold inflates macro AUC by
    # 0.087 (FINDINGS.md §9). Frequency is rounded to 2 dp because the raw value
    # is near-unique per study, which would make the grouping fake.
    headers = pd.read_parquet(headers_dir / "series_headers.parquet")
    frequency = pd.to_numeric(headers.get("imaging_frequency"), errors="coerce")
    headers = headers.assign(imaging_frequency_rounded=frequency.round(2))
    fields = ["manufacturer", "model_name", "software_versions", "field_strength",
              "imaging_frequency_rounded", "transmit_coil"]
    headers["fingerprint"] = headers[fields].astype("string").fillna("?").agg("|".join, axis=1)
    groups = headers.groupby("StudyInstanceUID").fingerprint.agg(
        lambda s: s.value_counts().index[0])

    # sorted() is load-bearing: it is what makes the ordering, and therefore the
    # fold split, reproducible across kernels and across runs.
    studies = sorted(available & set(soft.index) & set(groups.index))
    say(f"usable studies: {len(studies):,}")
    if len(studies) < min_studies:
        raise SystemExit(f"only {len(studies)} usable studies (min {min_studies}); "
                         "the cache mount is probably wrong — build the cache first")

    targets = soft.loc[studies, findings].astype(float).to_numpy()
    channels = soft.loc[studies, [f"{f}__channel" for f in findings]].to_numpy()
    masks = np.ones_like(targets, dtype=np.float32)
    if abstain_masks_loss:
        masks[channels == "absent"] = 0.0

    # Per-finding confidence, when the labeler recorded it. The loss already
    # multiplies by `mask`, so a continuous mask IS a confidence weight — an
    # explicit "severe" contributes fully, a hedge contributes less, without any
    # change to the loss itself. Older label files have no such column and are
    # unaffected, which is what keeps this comparable to the runs before it.
    confidence_columns = [f"{f}__weight" for f in findings]
    if all(column in soft.columns for column in confidence_columns):
        confidence = soft.loc[studies, confidence_columns].to_numpy(dtype=np.float32)
        masks = masks * np.nan_to_num(confidence, nan=1.0)
        say(f"per-finding confidence weights in use "
            f"(mean {float(masks[masks > 0].mean()):.3f} where supervised)")
    else:
        say("no per-finding confidence column; every supervised target counts equally")

    targets = np.nan_to_num(targets, nan=0.0).astype(np.float32)

    # Gold studies: every finding populated in train.csv. Weighted up because
    # they are the only targets known to match what the leaderboard scores.
    weights = np.ones(len(studies), dtype=np.float32)
    is_gold = np.zeros(len(studies), bool)
    if competition is not None:
        train_csv = pd.read_csv(competition / "train.csv").set_index("StudyInstanceUID")
        gold = train_csv[train_csv[findings].notna().all(axis=1)].index
        is_gold = np.array([s in set(gold) for s in studies])
        weights[is_gold] = gold_weight
        for position, study in enumerate(studies):
            if is_gold[position]:
                targets[position] = train_csv.loc[study, findings].to_numpy(dtype=np.float32)
                masks[position] = 1.0
        say(f"gold studies in cache: {int(is_gold.sum())} (weight {gold_weight})")

    group_values = groups.loc[studies].to_numpy()
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(studies)), None, group_values))

    return {"studies": studies, "cache_paths": cache_paths, "targets": targets,
            "masks": masks, "weights": weights, "is_gold": is_gold,
            "group_values": group_values, "splits": splits}
