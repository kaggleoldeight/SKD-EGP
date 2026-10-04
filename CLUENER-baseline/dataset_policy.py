"""CLUENER dataset-specific, evidence-backed augmentation policy.

The weak-class set follows the same rule as CMeEE-V2: use the lower half of
per-class F1 values at the epoch with the highest reconstructed overall Dev
micro-F1.  Epoch indices in the supplied CSV are zero-based.
"""

DATASET_NAME = "CLUENER"
FOCUS_CLASS_SELECTION_POLICY = (
    "bottom_half_by_class_f1_at_overall_best_dev_micro_f1_epoch"
)
BASELINE_EVIDENCE_FILE = "CLUENER_train_class_log_baseline.csv"
BASELINE_EVIDENCE_SHA256 = (
    "8b7964ac96a1f8454ab64255223c50e4ae8950e928650774b4f0fa8d8bc94873"
)
BASELINE_BEST_EPOCH_INDEX = 5
BASELINE_BEST_EPOCH_NUMBER = 6
BASELINE_DEV_MICRO_F1 = 0.8025088452878739
BASELINE_DEV_COUNTS = {"tp": 2495, "pred": 3146, "gold": 3072}
BASELINE_CLASS_F1_AT_BEST_EPOCH = {
    "address": 0.649077,
    "book": 0.813333,
    "company": 0.818898,
    "game": 0.851266,
    "government": 0.839458,
    "movie": 0.781690,
    "name": 0.874735,
    "organization": 0.824632,
    "position": 0.806415,
    "scene": 0.724566,
}
FOCUS_CLASS_PRIORITY = ("address", "scene", "movie", "position", "book")
FOCUS_ENTITY_TYPES = frozenset(FOCUS_CLASS_PRIORITY)

