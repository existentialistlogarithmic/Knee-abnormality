"""Build a synthetic competition that mimics the *believed* structure.

Nothing here is a finding. It exists so the Phase 0 scripts can be exercised —
and the redaction guarantees tested — without Kaggle credentials and without
touching a single byte of patient data.

The shape it mimics (all UNVERIFIED, see docs/FINDINGS.md): ~4,400 studies,
12 binary findings populated for only a small expert-labelled subset, a
multilingual report table covering most but not all studies, and a site column.
"""

from __future__ import annotations

from pathlib import Path

FINDINGS = [
    "acl_tear", "pcl_tear", "meniscus_tear_medial", "meniscus_tear_lateral",
    "chondral_defect", "bone_marrow_edema", "joint_effusion", "bakers_cyst",
    "fracture", "mcl_injury", "lcl_injury", "patellar_maltracking",
]

REPORTS = {
    "en": "MRI of the right knee. No evidence of meniscal tear. Mild joint effusion "
          "is present within the suprapatellar recess.",
    "de": "MRT des linken Knies. Kein Nachweis einer Meniskusläsion. Geringer "
          "Gelenkerguss im Recessus suprapatellaris.",
    "fr": "IRM du genou droit. Pas de déchirure méniscale. Épanchement articulaire "
          "modéré du récessus suprapatellaire.",
    "es": "RM de rodilla izquierda. No se observa rotura del menisco. Derrame "
          "articular leve en el receso suprarrotuliano.",
    "tr": "Sol diz MR. Menisküs yırtığı izlenmedi. Suprapatellar reseste hafif "
          "eklem efüzyonu mevcuttur.",
    "ja": "左膝MRI。半月板断裂は認めない。膝蓋上嚢に少量の関節液貯留を認める。",
}

UID_PREFIX = "1.2.826.0.1.3680043."
N_STUDIES = 4407
N_GOLD = 58
N_REPORTS = 4380


def build(root: Path, data_dir: Path, out_dir: Path) -> dict:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    uids = [f"{UID_PREFIX}{10000 + i}" for i in range(N_STUDIES)]
    train = pd.DataFrame({"StudyInstanceUID": uids})
    for i, finding in enumerate(FINDINGS):
        col = np.full(N_STUDIES, np.nan)
        idx = rng.choice(N_STUDIES, N_GOLD, replace=False)
        col[idx] = rng.random(N_GOLD) < (0.05 + 0.03 * i)
        train[finding] = col
    train["laterality"] = rng.choice(["L", "R"], N_STUDIES)
    train["site_id"] = rng.choice([f"SITE_{i:02d}" for i in range(17)], N_STUDIES)
    train.to_csv(data_dir / "train.csv", index=False)

    keys = list(REPORTS)
    pd.DataFrame(
        {
            "StudyInstanceUID": uids[:N_REPORTS],
            "report_text": [REPORTS[keys[i % len(keys)]] * int(rng.integers(1, 5))
                            for i in range(N_REPORTS)],
            "report_language_declared": [keys[i % len(keys)] for i in range(N_REPORTS)],
        }
    ).to_csv(data_dir / "train_reports.csv", index=False)

    pd.DataFrame(
        {"StudyInstanceUID": uids[:900], **dict.fromkeys(FINDINGS, 0.5)}
    ).to_csv(data_dir / "sample_submission.csv", index=False)

    names = ["train.csv", "train_reports.csv", "sample_submission.csv"]
    inventory = pd.DataFrame(
        {
            "name": names + [f"train_images/{uids[0]}/series1/1.dcm"],
            "total_bytes": [(data_dir / n).stat().st_size for n in names] + [512_000],
            "creation_date": ["2026-06-01"] * 4,
        }
    )
    inventory.to_csv(out_dir / "competition_files.csv", index=False)
    return {"uids": uids, "reports": REPORTS, "findings": FINDINGS}
