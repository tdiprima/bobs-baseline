"""
Canonical feature list for order anomaly detection.

Single source of truth — all modules that need to select or transform
features reference this constant rather than maintaining their own copy.
"""

FEATURE_COLS = [
    "items_per_order",
    "time_since_last_order",
    "order_time",
    "customer_frequency",
    "total_cost",
    "prep_time_estimate",
]
