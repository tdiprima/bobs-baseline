"""
Test your own orders through Bob's trained detector.

Run from the project root:
    python scripts/test_orders.py
"""

import sys
from pathlib import Path

# Add project root to path so src.* imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import explain_deviations
from src.data_source import SyntheticDataSource
from src.features import FEATURE_COLS
from src.model import OrderAnomalyModel


def test_custom_order() -> None:
    """Interactive tool to test custom orders."""
    print("=" * 70)
    print("🍔 BOB'S BURGERS: CUSTOM ORDER TESTER")
    print("=" * 70)

    # Load trained model
    print("\n📦 Loading trained model...")
    data_source = SyntheticDataSource(n_samples=1000)
    df = data_source.load_orders()

    model = OrderAnomalyModel()
    df = model.fit(df)

    print("\n✅ Model loaded successfully!")
    print("\nNow let's test some interesting orders...\n")

    # Test cases
    test_cases = [
        {
            "name": "My Regular Lunch",
            "order": {
                "items_per_order": 3,          # normal-sized order
                "time_since_last_order": 720,  # ~12 hours
                "order_time": 13.0,            # normal lunch
                "customer_frequency": 0.85,    # frequent regular
                "total_cost": 32.50,           # reasonable spend
                "prep_time_estimate": 18,      # standard prep
            },
            "expected": "🟢 Looks normal; no action needed",
        },
    ]

    normal_stats = df[df["cluster"] != -1][FEATURE_COLS].mean()

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test['name']}")
        print("=" * 70)

        order = test["order"]

        # Display order details
        print("\n📋 Order Details:")
        print(f"   Items: {order['items_per_order']}")
        print(f"   Time since last: {order['time_since_last_order']} minutes")
        print(f"   Order time: {order['order_time']:.1f} (hour)")
        print(f"   Customer frequency: {order['customer_frequency']}")
        print(f"   Total cost: ${order['total_cost']:.2f}")
        print(f"   Prep time: {order['prep_time_estimate']} minutes")

        result = model.score(order)
        print(f"\n🎯 Model Prediction: {result['cluster_name']}")
        print(f"   Expected: {test['expected']}")

        if result["is_anomaly"]:
            print("\n🚨 BOB'S REACTION: 'Oh no... not again...'")
            print("   Status: ANOMALY DETECTED")
            print("   Recommendation: FLAG FOR REVIEW")
        else:
            print("\n😊 BOB'S REACTION: 'I can handle this'")
            print("   Status: Normal operation")

        # Feature deviation analysis
        print("\n📊 Feature Deviation from Normal:")
        for d in explain_deviations(order, normal_stats):
            print(f"   {d['severity']} {d['feature']}: {d['diff_pct']:+.1f}% from baseline")


if __name__ == "__main__":
    test_custom_order()
    print("\n" + "=" * 70)
    print("✅ Testing Complete!")
    print("=" * 70)
    print("\nGo forth and detect weird vibes! 🛡️")
