"""CMeEE-V2 dataset-specific, evidence-backed augmentation policy.

The weak-class set is the lower half of per-class F1 values at the epoch whose
reconstructed overall Dev micro-F1 is maximal.  Epoch indices in the supplied
CSV are zero-based.
"""

DATASET_NAME = "CMeEE-V2"
FOCUS_CLASS_SELECTION_POLICY = (
    "bottom_half_by_class_f1_at_overall_best_dev_micro_f1_epoch"
)
BASELINE_EVIDENCE_FILE = "train_class_log_CMeEE_baseline.csv"
BASELINE_EVIDENCE_SHA256 = (
    "386c802295a9dd0ec0597a1cabef3c256504a4f2aff97507ea0ec7707028a704"
)
BASELINE_BEST_EPOCH_INDEX = 4
BASELINE_BEST_EPOCH_NUMBER = 5
BASELINE_DEV_MICRO_F1 = 0.7434716981132076
BASELINE_DEV_COUNTS = {"tp": 19702, "pred": 26919, "gold": 26081}
BASELINE_CLASS_F1_AT_BEST_EPOCH = {
    "bod": 0.725196,
    "dis": 0.854183,
    "sym": 0.658332,
    "mic": 0.799682,
    "pro": 0.716600,
    "ite": 0.571958,
    "dep": 0.819149,
    "dru": 0.844527,
    "equ": 0.613811,
}
FOCUS_CLASS_PRIORITY = ("ite", "equ", "sym", "pro", "bod")
FOCUS_ENTITY_TYPES = frozenset(FOCUS_CLASS_PRIORITY)

