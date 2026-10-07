"""
visualize_network.py
---------------------
Quick visual sanity-check of the Complexity Science and Evolution Unit
component: draws the user-seller network before and after a severe
disruption + a few cascading-failure steps, coloring nodes by trust.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt

from network_model import (
    build_platform_network,
    inject_disruption,
    cascading_failure_step,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def draw_network(G, ax, title):
    pos = nx.spring_layout(G, seed=7)
    trust_values = [G.nodes[n]["trust"] for n in G.nodes]
    node_shapes = ["o" if G.nodes[n]["type"] == "user" else "s" for n in G.nodes]

    # draw users and sellers separately so we can use different marker shapes
    users = [n for n in G.nodes if G.nodes[n]["type"] == "user"]
    sellers = [n for n in G.nodes if G.nodes[n]["type"] == "seller"]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, nodelist=users, node_shape="o",
                            node_color=[G.nodes[n]["trust"] for n in users],
                            cmap=plt.cm.RdYlGn, vmin=0, vmax=1, node_size=80, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=sellers, node_shape="s",
                            node_color=[G.nodes[n]["trust"] for n in sellers],
                            cmap=plt.cm.RdYlGn, vmin=0, vmax=1, node_size=180, ax=ax)
    ax.set_title(title)
    ax.axis("off")


if __name__ == "__main__":
    G = build_platform_network(n_users=40, n_sellers=8, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw_network(G, axes[0], "Before disruption\n(circles=users, squares=sellers)")

    inject_disruption(G, severity=0.9, seed=42)
    for _ in range(4):
        cascading_failure_step(G)
    draw_network(G, axes[1], "After disruption + cascade\n(color = trust level)")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "network_before_after.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
