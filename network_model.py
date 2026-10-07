"""
network_model.py
-----------------
Complexity Science and Evolution Unit part of the project.

Models the digital platform as a complex network:
    - nodes  = users and sellers
    - edges  = transactions / interactions (weighted by frequency)
    - trust  = a per-node attribute that can degrade and cascade
               through the network when a disruption occurs

This module provides:
    build_platform_network(...)   -> networkx.Graph
    inject_disruption(...)        -> simulates an attack / outage on one node
    cascading_failure_step(...)   -> propagates trust loss through neighbors
    network_health(...)           -> scalar summary of the whole network state
"""

import random
import networkx as nx
import numpy as np


def build_platform_network(n_users=40, n_sellers=8, avg_edges_per_user=3, seed=None):
    """
    Builds a bipartite-like network: users connect to sellers they buy from.
    Users can also be weakly connected to each other (social/referral effect).

    Each node has:
        - type: 'user' or 'seller'
        - trust: float in [0, 1], starts high
        - active: bool, whether the node is currently functioning normally
    """
    rng = random.Random(seed)
    G = nx.Graph()

    for u in range(n_users):
        G.add_node(f"user_{u}", type="user", trust=1.0, active=True)
    for s in range(n_sellers):
        G.add_node(f"seller_{s}", type="seller", trust=1.0, active=True)

    user_nodes = [f"user_{u}" for u in range(n_users)]
    seller_nodes = [f"seller_{s}" for s in range(n_sellers)]

    # users buy from a handful of sellers, weighted by a random popularity score
    seller_popularity = rng.choices(range(1, 10), k=n_sellers)
    for u in user_nodes:
        n_links = max(1, int(rng.gauss(avg_edges_per_user, 1)))
        chosen_sellers = rng.choices(
            seller_nodes, weights=seller_popularity, k=min(n_links, n_sellers)
        )
        for s in set(chosen_sellers):
            weight = rng.randint(1, 5)  # transaction frequency/volume
            G.add_edge(u, s, weight=weight)

    # light social layer between users (referrals, shared reviews)
    for _ in range(n_users // 2):
        a, b = rng.sample(user_nodes, 2)
        if not G.has_edge(a, b):
            G.add_edge(a, b, weight=rng.randint(1, 2))

    return G


def inject_disruption(G, target=None, severity=0.7, seed=None):
    """
    Simulates an attack, outage, or sudden loss of trust at one node.
    Returns the name of the disrupted node.
    """
    rng = random.Random(seed)
    if target is None:
        target = rng.choice(list(G.nodes))
    G.nodes[target]["trust"] = max(0.0, G.nodes[target]["trust"] - severity)
    if G.nodes[target]["trust"] < 0.2:
        G.nodes[target]["active"] = False
    return target


def cascading_failure_step(G, spread_factor=0.15):
    """
    Propagates trust loss one hop outward from low-trust nodes,
    weighted by edge strength (stronger ties transmit more distrust).
    This is a simple discrete-time contagion model, not a physical simulation.
    """
    updates = {}
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue
        trust_pressure = 0.0
        total_weight = 0.0
        for nb in neighbors:
            w = G[node][nb].get("weight", 1)
            trust_pressure += (1.0 - G.nodes[nb]["trust"]) * w
            total_weight += w
        if total_weight > 0:
            avg_pressure = trust_pressure / total_weight
            new_trust = G.nodes[node]["trust"] - spread_factor * avg_pressure
            updates[node] = max(0.0, min(1.0, new_trust))

    for node, new_trust in updates.items():
        G.nodes[node]["trust"] = new_trust
        if new_trust < 0.2:
            G.nodes[node]["active"] = False

    return G


def network_health(G):
    """
    Scalar summary of overall network condition, in [0, 1].
    Combines average trust and fraction of still-active nodes.
    """
    trusts = [d["trust"] for _, d in G.nodes(data=True)]
    active = [1 if d["active"] else 0 for _, d in G.nodes(data=True)]
    avg_trust = float(np.mean(trusts))
    active_fraction = float(np.mean(active))
    return 0.5 * avg_trust + 0.5 * active_fraction


def reset_network(G):
    """Restores all nodes to full trust / active state (used between RL episodes)."""
    for node in G.nodes:
        G.nodes[node]["trust"] = 1.0
        G.nodes[node]["active"] = True
    return G
