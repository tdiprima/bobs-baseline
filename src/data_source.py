"""
DataSource adapters for order ingestion.

SyntheticDataSource — generates Bob's Burgers order data for development/testing.
CsvDataSource       — loads real order data from a CSV file.

Real-world analogy: historical API requests, job logs, transaction records.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(ABC):
    @abstractmethod
    def load_orders(self) -> pd.DataFrame:
        """Load order data. Returns a DataFrame with the standard order schema."""
        ...


class SyntheticDataSource(DataSource):
    """Generates synthetic Bob's Burgers order data."""

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed

    def load_orders(self) -> pd.DataFrame:
        np.random.seed(self.seed)

        data = []
        current_time = datetime(2024, 1, 1, 8, 0)

        # Customer profiles (like different user behavior patterns)
        customers = {
            "teddy": {"frequency": 0.95, "pattern": "regular"},
            "linda": {"frequency": 0.80, "pattern": "chaos"},
            "tina": {"frequency": 0.70, "pattern": "normal"},
            "gene": {"frequency": 0.65, "pattern": "normal"},
            "louise": {"frequency": 0.60, "pattern": "weird"},
            "hugo": {"frequency": 0.30, "pattern": "timing"},
            "random_customer": {"frequency": 0.10, "pattern": "normal"},
        }

        for i in range(self.n_samples):
            # Pick customer type (weighted)
            customer_weights = [0.15, 0.12, 0.15, 0.15, 0.10, 0.05, 0.28]
            customer = np.random.choice(list(customers.keys()), p=customer_weights)
            profile = customers[customer]

            # Generate order based on profile
            if profile["pattern"] == "regular":
                # Teddy: Same thing every day (normal baseline)
                items = np.random.randint(1, 3)
                cost = items * np.random.uniform(8, 12)
                prep_time = items * 5
                time_since_last = np.random.uniform(1400, 1500)  # ~daily

            elif profile["pattern"] == "chaos":
                # Linda: Chaotic but harmless
                items = np.random.randint(2, 8)
                cost = items * np.random.uniform(7, 15)
                prep_time = items * np.random.uniform(4, 8)
                time_since_last = np.random.uniform(180, 600)

            elif profile["pattern"] == "timing":
                # Hugo: Shows up at bad times
                items = np.random.randint(1, 4)
                cost = items * np.random.uniform(9, 13)
                prep_time = items * 6
                time_since_last = np.random.uniform(2000, 5000)
                # Force late hours
                current_time = current_time.replace(hour=np.random.choice([21, 22, 23]))

            elif profile["pattern"] == "weird":
                # Louise: Suspicious but not quite anomalous
                items = np.random.randint(3, 7)
                cost = items * np.random.uniform(6, 14)
                prep_time = items * np.random.uniform(5, 9)
                time_since_last = np.random.uniform(300, 900)

            else:
                # Normal customers
                items = np.random.randint(1, 5)
                cost = items * np.random.uniform(8, 12)
                prep_time = items * 5
                time_since_last = np.random.uniform(500, 3000)

            # Add some true anomalies (5% of data)
            if np.random.random() < 0.05:
                # "30 burgers in 2 minutes" scenarios
                anomaly_type = np.random.choice(
                    ["massive", "rapid", "cost_spike", "timing_weird"]
                )

                if anomaly_type == "massive":
                    items = np.random.randint(25, 50)
                    cost = items * np.random.uniform(8, 12)
                    prep_time = items * 5

                elif anomaly_type == "rapid":
                    items = np.random.randint(8, 15)
                    time_since_last = np.random.uniform(1, 30)  # Too fast
                    cost = items * np.random.uniform(8, 12)
                    prep_time = items * 5

                elif anomaly_type == "cost_spike":
                    items = np.random.randint(3, 8)
                    cost = items * np.random.uniform(50, 150)  # Way too expensive
                    prep_time = items * 5

                elif anomaly_type == "timing_weird":
                    items = np.random.randint(15, 30)
                    current_time = current_time.replace(
                        hour=np.random.choice([3, 4, 23])
                    )
                    cost = items * np.random.uniform(8, 12)
                    prep_time = items * 5

            # Simulate time progression
            current_time += timedelta(minutes=np.random.randint(5, 45))

            data.append(
                {
                    "order_id": f"ORD_{i:05d}",
                    "customer": customer,
                    "items_per_order": items,
                    "time_since_last_order": time_since_last,
                    "order_time": current_time.hour + current_time.minute / 60,
                    "customer_frequency": profile["frequency"],
                    "total_cost": cost,
                    "prep_time_estimate": prep_time,
                    "timestamp": current_time,
                }
            )

        df = pd.DataFrame(data)
        logger.info("Generated %d synthetic orders", len(df))
        return df


class CsvDataSource(DataSource):
    """Loads real order data from a CSV file."""

    def __init__(self, path: str):
        self._path = path

    def load_orders(self) -> pd.DataFrame:
        df = pd.read_csv(self._path)
        logger.info("Loaded %d orders from %s", len(df), self._path)
        return df
