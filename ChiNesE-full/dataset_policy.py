"""ChiNesE dataset-specific, evidence-backed augmentation policy.

The weak-class set follows the same rule as CMeEE-V2: use the lower half of
per-class F1 values at the epoch with the highest reconstructed overall Dev
micro-F1.  Epoch indices in the supplied CSV are zero-based.
"""

DATASET_NAME = "ChiNesE"
FOCUS_CLASS_SELECTION_POLICY = (
    "bottom_half_by_class_f1_at_overall_best_dev_micro_f1_epoch"
)
BASELINE_EVIDENCE_FILE = "train_class_log_ChiNesE_baseline.csv"
BASELINE_EVIDENCE_SHA256 = (
    "5dbbf94e3a90e2a267d6166b38e128e04ea905c0c4a9f66b57a1a673c5f9e460"
)
BASELINE_BEST_EPOCH_INDEX = 4
BASELINE_BEST_EPOCH_NUMBER = 5
BASELINE_DEV_MICRO_F1 = 0.8501950268161872
BASELINE_DEV_COUNTS = {"tp": 6975, "pred": 7576, "gold": 8832}
BASELINE_CLASS_F1_AT_BEST_EPOCH = {
    "Person": 0.934771,
    "Location": 0.889971,
    "Organization": 0.838049,
    "Event": 0.742857,
    "Work": 0.748850,
    "Time": 0.902935,
    "Product": 0.628366,
    "Food": 0.808722,
    "Medicine": 0.808036,
    "Creature": 0.727273,
}
FOCUS_CLASS_PRIORITY = ("Product", "Creature", "Event", "Work", "Medicine")
FOCUS_ENTITY_TYPES = frozenset(FOCUS_CLASS_PRIORITY)

