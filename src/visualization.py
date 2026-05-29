"""
Visualization for anomaly analysis output.

In production these would be your dashboards/alerts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from .features import FEATURE_COLS
from .model import CLUSTER_NAMES, OrderAnomalyModel

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

CLUSTER_COLORS: dict[int, str] = {
    -1: "#FF4444",  # Red for anomalies
    0: "#44AA44",   # Green for Teddy
    1: "#FFAA44",   # Orange for Linda
    2: "#4444FF",   # Blue for Hugo
}


def visualize(
    df,
    model: OrderAnomalyModel,
    output_path: str = "../outputs/anomaly_analysis.png",
):
    """Create visualizations. In production these would be dashboards/alerts."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "🍔 Bob's Burgers: Is This Order Gonna Be a Problem?",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Items vs Cost (classic scatter)
    ax = axes[0, 0]
    for cluster in df["cluster"].unique():
        mask = df["cluster"] == cluster
        label = CLUSTER_NAMES.get(cluster, f"Cluster {cluster}")
        ax.scatter(
            df[mask]["items_per_order"],
            df[mask]["total_cost"],
            c=CLUSTER_COLORS.get(cluster, "#888888"),
            label=label,
            alpha=0.6,
            s=50,
        )
    ax.set_xlabel("Items per Order")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Order Size vs Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Time patterns
    ax = axes[0, 1]
    for cluster in df["cluster"].unique():
        mask = df["cluster"] == cluster
        ax.scatter(
            df[mask]["order_time"],
            df[mask]["items_per_order"],
            c=CLUSTER_COLORS.get(cluster, "#888888"),
            alpha=0.6,
            s=50,
        )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Items per Order")
    ax.set_title("Timing Patterns")
    ax.grid(True, alpha=0.3)

    # 3. Frequency patterns
    ax = axes[0, 2]
    for cluster in df["cluster"].unique():
        mask = df["cluster"] == cluster
        ax.scatter(
            df[mask]["time_since_last_order"],
            df[mask]["items_per_order"],
            c=CLUSTER_COLORS.get(cluster, "#888888"),
            alpha=0.6,
            s=50,
        )
    ax.set_xlabel("Time Since Last Order (min)")
    ax.set_ylabel("Items per Order")
    ax.set_title("Request Frequency Patterns")
    ax.grid(True, alpha=0.3)

    # 4. Cost distribution
    ax = axes[1, 0]
    anomalies_cost = df[df["cluster"] == -1]["total_cost"]
    normal_cost = df[df["cluster"] != -1]["total_cost"]
    ax.hist(
        [normal_cost, anomalies_cost],
        bins=30,
        label=["Normal", "Anomalies"],
        color=["#44AA44", "#FF4444"],
        alpha=0.7,
    )
    ax.set_xlabel("Total Cost ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Cost Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. PCA visualization (2D projection of all features)
    ax = axes[1, 1]
    X = df[FEATURE_COLS].values
    X_scaled = model.scaler.transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    for cluster in df["cluster"].unique():
        mask = (df["cluster"] == cluster).values
        label = CLUSTER_NAMES.get(cluster, f"Cluster {cluster}")
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=CLUSTER_COLORS.get(cluster, "#888888"),
            label=label,
            alpha=0.6,
            s=50,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA: All Features Projection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Anomaly severity heatmap
    ax = axes[1, 2]
    anomalies = df[df["cluster"] == -1]
    if len(anomalies) > 0:
        severity_features = ["items_per_order", "total_cost", "prep_time_estimate"]
        severity_data = anomalies[severity_features].head(10)

        # Normalize for visualization
        severity_normalized = (severity_data - severity_data.min()) / (
            severity_data.max() - severity_data.min()
        )

        im = ax.imshow(severity_normalized.T, cmap="Reds", aspect="auto")
        ax.set_yticks(range(len(severity_features)))
        ax.set_yticklabels([f.replace("_", " ").title() for f in severity_features])
        ax.set_xlabel("Anomaly Sample")
        ax.set_title("Anomaly Feature Intensity")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📈 Visualization saved to: {output_path}")

    return fig
