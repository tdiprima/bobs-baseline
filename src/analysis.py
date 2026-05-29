"""
Anomaly analysis and feature deviation reporting.

This is what you'd present to your security/ops team.
"""

import pandas as pd

from .features import FEATURE_COLS

METRIC_LABELS = {
    "items_per_order": "Order Size",
    "time_since_last_order": "Time Since Last (min)",
    "total_cost": "Total Cost ($)",
    "prep_time_estimate": "Prep Time (min)",
    "order_time": "Order Hour",
}


def analyze_anomalies(df: pd.DataFrame) -> None:
    """Deep dive into what makes anomalies anomalous."""
    anomalies = df[df["cluster"] == -1]
    normal = df[df["cluster"] != -1]

    print("\n🚨 Anomaly Analysis - 'Would Bob Sigh?' Report")
    print("=" * 60)

    for col, label in METRIC_LABELS.items():
        normal_mean = normal[col].mean()
        anomaly_mean = anomalies[col].mean()
        diff_pct = ((anomaly_mean - normal_mean) / normal_mean) * 100

        print(f"\n{label}:")
        print(f"   Normal:    {normal_mean:.2f}")
        print(f"   Anomalies: {anomaly_mean:.2f}")
        print(f"   Diff:      {diff_pct:+.1f}%")

    # Show some example anomalies
    print("\n🔍 Sample 'Absolutely Not' Orders:")
    print("-" * 60)
    sample_anomalies = anomalies.sample(min(5, len(anomalies)))
    for _, row in sample_anomalies.iterrows():
        print(f"\n   Order {row['order_id']}:")
        print(f"      Customer: {row['customer']}")
        print(f"      Items: {row['items_per_order']}")
        print(f"      Cost: ${row['total_cost']:.2f}")
        print(f"      Time: {row['order_time']:.1f} (hour)")
        print("      Bob's reaction: 😰")


def explain_deviations(order: dict, normal_stats: pd.Series) -> list[dict]:
    """
    Compute per-feature deviation from baseline for a single order.

    Returns a list of dicts: feature, value, normal_val, diff_pct, severity.
    """
    deviations = []
    for feature in FEATURE_COLS:
        value = order[feature]
        normal_val = normal_stats[feature]
        diff_pct = ((value - normal_val) / normal_val) * 100

        if abs(diff_pct) > 100:
            severity = "🔴 EXTREME"
        elif abs(diff_pct) > 50:
            severity = "🟡 HIGH"
        else:
            severity = "🟢 NORMAL"

        deviations.append({
            "feature": feature,
            "value": value,
            "normal_val": normal_val,
            "diff_pct": diff_pct,
            "severity": severity,
        })
    return deviations
