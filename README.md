# 🍔 Bob's Baseline

Unsupervised anomaly detection engine that flags unusual behavioral patterns — built on a fake Bob's Burgers dataset, designed to map directly to production problems like API abuse, cost spikes, and resource exhaustion.

## When "Normal" Is Hard to Define

Rule-based alerting breaks down when normal isn't constant. A single large order is fine on a Friday night and suspicious at 3 AM. A burst of rapid requests is expected from a batch job and alarming from a new user. Static thresholds generate noise; they don't catch what actually feels wrong.

The same problem appears everywhere: detecting anomalous API calls, flagging cloud cost outliers, catching fraudulent transactions, identifying queue jobs that will blow up your infrastructure. You know it when you see it — but defining it upfront is hard.

## Density-Based Pattern Learning, No Labels Required

This project trains a DBSCAN model on order history to learn what "normal" looks like across six behavioral features. Anything that doesn't fit a dense cluster gets labeled an anomaly. No labeled training data, no manually tuned thresholds — the model finds the structure in the data.

Once trained, new orders are scored in O(log n) using a NearestNeighbors index over DBSCAN core samples, so the model never needs to re-fit on every request. Each flagged order comes with a per-feature deviation report that explains *why* it was anomalous — not just that it was.

The Bob's Burgers framing is intentional. The feature mapping is direct:

| Order Feature | Production Equivalent |
|---|---|
| `items_per_order` | request payload size / batch size |
| `time_since_last_order` | request interval / rate |
| `total_cost` | compute or cloud spend |
| `prep_time_estimate` | estimated processing duration |
| `order_time` | time-of-day usage pattern |
| `customer_frequency` | historical user activity |

## Example

A new order arrives: 35 items, $380 total, placed at 10:30 PM, only 15 minutes after the last order.

```
📥 New Order Received:
   Items: 35
   Cost: $380
   Time: 22.5 (10:30 PM)
   Last order: 15 min ago

🎯 Model Prediction: Absolutely Not 🚨

🚨 ALERT: Anomalous Order Detected!
   Recommended Action: Flag for review

📊 Why this classification?
   🔴 EXTREME items_per_order: 35.00 (normal: 3.12, diff: +1022.4%)
   🔴 EXTREME total_cost: 380.00 (normal: 28.47, diff: +1234.7%)
   🟢 NORMAL order_time: 22.50 (normal: 14.31, diff: +57.2%)
```

In a real system, this triggers rate limiting, an on-call alert, and a log entry for investigation.

## Usage

**Requirements:** Python 3.11+

```bash
pip install -e .
```

**Run the full pipeline** — generates synthetic orders, trains the model, analyzes anomalies, saves a visualization, and demos production scoring:

```bash
python -m src.bobs_burgers_anomaly
```

**Test custom orders** against a trained model:

```bash
python scripts/test_orders.py
```

**Use your own data** via `CsvDataSource`:

```python
from src.data_source import CsvDataSource
from src.model import OrderAnomalyModel

data_source = CsvDataSource("path/to/orders.csv")
df = data_source.load_orders()

model = OrderAnomalyModel()
df = model.fit(df)

result = model.score({
    "items_per_order": 35,
    "time_since_last_order": 15,
    "order_time": 22.5,
    "customer_frequency": 0.05,
    "total_cost": 380,
    "prep_time_estimate": 175,
})

print(result["is_anomaly"])    # True
print(result["cluster_name"])  # "Absolutely Not 🚨"
```

**Tune the model** via DBSCAN parameters:

```python
model = OrderAnomalyModel(eps=0.8, min_samples=10)
```

- `eps` — neighborhood radius; smaller catches subtler outliers, larger reduces noise
- `min_samples` — minimum points to form a core cluster; higher requires denser normal patterns

Output artifacts are written to `outputs/anomaly_analysis.png` (six-panel visualization: scatter plots, PCA projection, cost distribution, anomaly intensity heatmap) and `data/bobs_burgers_orders.csv`.

<br>
