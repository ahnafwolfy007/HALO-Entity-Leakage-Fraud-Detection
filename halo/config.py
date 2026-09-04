"""Single source of truth for every tunable in the HALO pipeline.

Nothing anywhere else in the package may hard-code a hyperparameter, a seed, or a
path. If you need a knob, add it here. `config_hash()` is written into every report
so a result can always be traced back to the exact configuration that produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths. On Kaggle these resolve to /kaggle/input and /kaggle/working automatically.
# --------------------------------------------------------------------------------------

ON_KAGGLE = Path("/kaggle/input").exists()

if ON_KAGGLE:
    INPUT_DIR = Path("/kaggle/input")
    WORK_DIR = Path("/kaggle/working")
    IEEE_DIR = INPUT_DIR / "ieee-fraud-detection"
    PAYSIM_CSV = INPUT_DIR / "paysim1" / "PS_20174392719_1491204439457_log.csv"
else:
    _here = Path(__file__).resolve().parent.parent
    INPUT_DIR = _here / "data"
    WORK_DIR = _here / "out"
    IEEE_DIR = INPUT_DIR / "ieee-fraud-detection"
    PAYSIM_CSV = INPUT_DIR / "paysim" / "paysim.csv"

CKPT_DIR = WORK_DIR / "checkpoints"
RESULTS_DIR = WORK_DIR / "results"
FIG_DIR = WORK_DIR / "figures"

for _d in (WORK_DIR, CKPT_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------------------

@dataclass
class Config:
    # ---- reproducibility -------------------------------------------------------------
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    """>=5 seeds. Every headline number is reported mean +/- std across these."""

    # ---- protocol --------------------------------------------------------------------
    fold_days: int = 30
    """Length of each rolling-origin test fold, in days."""

    min_train_days: int = 60
    """Days of history required before the first test fold opens."""

    latency_deltas: tuple[int, ...] = (0, 7, 30, 60, 120)
    """Label-maturity gate sweep, in days (T5)."""

    headline_delta: int = 30
    """Realistic card-not-present chargeback lag. Used for T1/T3 headline numbers."""

    strict_delta: int = 120
    """Vesta's stated window. Reported on a reduced test window."""

    # ---- Block A: entity resolution ---------------------------------------------------
    uid_variant: str = "medium"
    """One of strict | medium | loose. All three are run for the sensitivity table."""

    fs_agreement_threshold: float = 0.55
    """Fellegi-Sunter weighted agreement score above which two blocks may merge."""

    fs_weights: dict = field(default_factory=lambda: {
        "card2": 1.0, "card3": 0.6, "card5": 0.6, "card6": 0.4, "card4": 0.4,
        "P_emaildomain": 1.2, "dist1": 0.5,
        "DeviceInfo": 1.4, "id_30": 0.8, "id_31": 0.8, "id_33": 1.0,
    })

    # ---- Block B: association risk ----------------------------------------------------
    risk_attributes: tuple[str, ...] = (
        "card1", "card2", "addr1", "P_emaildomain", "R_emaildomain",
        "DeviceInfo", "id_31", "id_33", "ProductCD_card4", "bin_proxy",
    )
    tau_days: float = 30.0
    """Exponential time-decay constant for the Beta posterior."""

    gamma_damping: float = 0.4
    """Damping on the one-hop entity -> shared-attribute -> entity propagation."""

    eb_min_count: int = 30
    """Minimum observations for an attribute value to contribute to the EB prior fit."""

    # ---- Block C: missingness regimes -------------------------------------------------
    regime_min_support: float = 0.005
    """Minimum support for a co-missingness pattern to become its own regime."""

    regime_max_count: int = 60
    """Cap on the number of regimes; the rest are folded into nearest neighbours."""

    # ---- Block D: model ---------------------------------------------------------------
    lgbm_params: dict = field(default_factory=lambda: {
        # Deliberately conservative. A `num_leaves=63, min_child_samples=50` config was
        # tried first and overfit badly at this pipeline's actual per-fold sample sizes
        # (~6-9k rows after entity-disjoint splitting): T4's ablation ranking flipped
        # completely under a regularization change alone, with deltas smaller than their
        # own seed-to-seed std -- proof the ranking was noise, not signal. These values
        # were chosen to keep that from recurring; retune via the inner search
        # (`--tuning-budget`) rather than loosening this back up by hand.
        "objective": "binary",
        "n_estimators": 300,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_lambda": 3.0,
        "n_jobs": -1,
        "verbose": -1,
    })

    fn_cost_ratio: float = 20.0
    """Cost of a missed fraud relative to the cost of investigating a false positive."""

    monotone_features_positive: tuple[str, ...] = ()
    """Filled at runtime by model.py from the Block-B feature names."""

    # ---- evaluation -------------------------------------------------------------------
    review_budget_k: float = 0.01
    """Manual-review capacity as a fraction of transactions, for Dollar-Recall@k."""

    # ---- runtime ----------------------------------------------------------------------
    smoke: bool = False
    smoke_rows: int = 50_000
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config_hash"] = self.config_hash()
        return d


CFG = Config()


# --------------------------------------------------------------------------------------
# Leakage ladder rung definitions (T1). Each rung closes one more channel.
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Rung:
    key: str
    label: str
    random_split: bool          # True = leaky random shuffle; False = chronological
    resample_before_split: bool  # True = L1 open (SMOTE fitted across the split)
    entity_disjoint: bool        # True = L3 closed
    latency_delta: int           # 0 = L4 open
    past_only_encoding: bool     # False = L2 open (encodings fitted over full period)


LADDER: tuple[Rung, ...] = (
    Rung("rung0", "All leaks open (random split + pre-split SMOTE)",
         random_split=True, resample_before_split=True,
         entity_disjoint=False, latency_delta=0, past_only_encoding=False),
    Rung("rung1", "Close L1 (resample inside folds only)",
         random_split=True, resample_before_split=False,
         entity_disjoint=False, latency_delta=0, past_only_encoding=False),
    Rung("rung2", "Close L2 (chronological split, past-only encodings)",
         random_split=False, resample_before_split=False,
         entity_disjoint=False, latency_delta=0, past_only_encoding=True),
    Rung("rung3", "Close L3 (entity-disjoint train/test)",
         random_split=False, resample_before_split=False,
         entity_disjoint=True, latency_delta=0, past_only_encoding=True),
    Rung("rung4", "Close L4 (only labels matured past delta)",
         random_split=False, resample_before_split=False,
         entity_disjoint=True, latency_delta=CFG.headline_delta, past_only_encoding=True),
)
