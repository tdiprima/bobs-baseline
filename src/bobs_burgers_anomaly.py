"""
🍔 Is This Order Gonna Be a Problem?
Bob's Burgers Order Anomaly Detection

Maps to real-world: API abuse, cost anomalies, resource exhaustion, fraud detection

Run from the project root:
    python -m src.bobs_burgers_anomaly
"""

import warnings
from pathlib import Path

from .analysis import analyze_anomalies, explain_deviations
from .data_source import DataSource, SyntheticDataSource
from .features import FEATURE_COLS
from .model import CLUSTER_NAMES, OrderAnomalyModel
from .visualization import visualize

_PROJECT_ROOT = Path(__file__).parent.parent

warnings.filterwarnings("ignore")


def run_pipeline(data_source: DataSource) -> tuple:
    """
    Load data, train the model, and return (df, model).

    Accepts any DataSource adapter so callers can switch between synthetic
    and real order data without touching pipeline logic.
    """
    df = data_source.load_orders()
    print("🍔 Loaded Bob's Burgers Order Data")
    print(f"   Total orders: {len(df)}")
    print(f"   Features: {len(FEATURE_COLS)}")

    model = OrderAnomalyModel()
    df = model.fit(df)

    anomaly_count = (df["cluster"] == -1).sum()
    anomaly_pct = (anomaly_count / len(df)) * 100

    print("\n📊 Training Complete")
    print(f"   Clusters found: {df['cluster'].nunique()}")
    print(f"   Anomalies detected: {anomaly_count} ({anomaly_pct:.1f}%)")
    print("\n   Cluster Distribution:")
    for cluster_id, count in df["cluster"].value_counts().sort_index().items():
        name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        print(f"      {name}: {count} orders ({count/len(df)*100:.1f}%)")

    return df, model


def demo_production_scoring(df, model: OrderAnomalyModel) -> None:
    """
    Show how this would work in production.
    Real-world: This would be your API endpoint or batch job.
    """
    print("\n🔧 Production Deployment Pattern")
    print("=" * 60)

    # Simulate new incoming order
    new_order = {
        "items_per_order": 35,
        "time_since_last_order": 15,
        "order_time": 22.5,
        "customer_frequency": 0.05,
        "total_cost": 380,
        "prep_time_estimate": 175,
    }

    print("\n📥 New Order Received:")
    print(f"   Items: {new_order['items_per_order']}")
    print(f"   Cost: ${new_order['total_cost']}")
    print(f"   Time: {new_order['order_time']:.1f} (10:30 PM)")
    print(f"   Last order: {new_order['time_since_last_order']} min ago")

    result = model.score(new_order)
    print(f"\n🎯 Model Prediction: {result['cluster_name']}")

    if result["is_anomaly"]:
        print("\n🚨 ALERT: Anomalous Order Detected!")
        print("   Recommended Action: Flag for review")
        print("   Bob's status: 😰 'Oh no...'")
        print("\n   Real-world mapping:")
        print("      → Trigger rate limiting")
        print("      → Send alert to on-call")
        print("      → Log for investigation")
        print("      → Potentially reject/throttle")
    else:
        print("\n✅ Normal Order Pattern")
        print("   Bob's status: 😊 'I got this'")

    # Feature importance for this prediction
    normal_stats = df[df["cluster"] != -1][FEATURE_COLS].mean()
    deviations = explain_deviations(new_order, normal_stats)

    print("\n📊 Why this classification?")
    for d in deviations:
        print(
            f"   {d['severity']} {d['feature']}: {d['value']:.2f}"
            f" (normal: {d['normal_val']:.2f}, diff: {d['diff_pct']:+.1f}%)"
        )


def main() -> None:
    """Run the complete Bob's Burgers anomaly detection pipeline."""
    print("=" * 70)
    print("🍔 BOB'S BURGERS ANOMALY DETECTION SYSTEM")
    print("   'Is This Order Gonna Be a Problem?'")
    print("=" * 70)

    # Step 1 + 2: Load data and train
    print("\n📦 Step 1: Generate Synthetic Order Data")
    print("-" * 70)
    df, model = run_pipeline(SyntheticDataSource(n_samples=1000))

    # Step 3: Analyze anomalies
    print("\n🔍 Step 3: Analyze Detected Anomalies")
    print("-" * 70)
    analyze_anomalies(df)

    # Step 4: Visualize
    print("\n📊 Step 4: Generate Visualizations")
    print("-" * 70)
    output_png = str(_PROJECT_ROOT / "outputs" / "anomaly_analysis.png")
    visualize(df, model, output_path=output_png)

    # Step 5: Production scoring demo
    print("\n🚀 Step 5: Production Scoring Demo")
    print("-" * 70)
    demo_production_scoring(df, model)

    # Save data for further analysis
    output_csv = str(_PROJECT_ROOT / "data" / "bobs_burgers_orders.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Data saved to: {output_csv}")

    # Blue-team mapping summary
    print("\n" + "=" * 70)
    print("🛡️  BLUE-TEAM PRODUCTION MAPPING")
    print("=" * 70)
    print(
        """
This exact pattern applies to:

1. API Abuse Detection
   items_per_order → request_payload_size
   time_since_last → request_interval
   total_cost → compute_resources_used

2. Cost Anomaly Detection
   total_cost → cloud_spending
   prep_time → processing_duration
   customer_frequency → service_usage_pattern

3. Queue/Job Monitoring
   items_per_order → job_size
   prep_time → estimated_duration
   order_time → scheduled_time

4. Resource Exhaustion Precursors
   All features → early warning signals
   Anomaly detection → proactive alerting

The model asks: "Does this feel like one of THOSE requests?"
Not "Is this malicious?" — that's a different problem.

This is about: vibes, patterns, "hmm that's weird" detection.
"""
    )

    print("\n✅ Pipeline Complete!")
    print("   Check outputs/ and data/ for results")


if __name__ == "__main__":
    main()
