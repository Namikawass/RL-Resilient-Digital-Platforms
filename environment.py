"""
environment.py
---------------
Bridges the Neural Computation Unit (RL agent) and the
Complexity Science and Evolution Unit (network model).

The platform is represented as a small Markov Decision Process:

    state  -> discretized summary of the network (health, active seller
              fraction, disruption flag)
    action -> one of a small set of platform-management actions
    reward -> R = alpha*U + beta*T - gamma*C - delta*S - lambda*P
              (utility, economic throughput, computational cost,
               systemic risk, privacy penalty)

This is intentionally simple (tabular, discretized) so it stays
transparent and runnable without heavy ML dependencies -- the point
of the prototype is to demonstrate the pipeline, not to be
production-grade.
"""

import numpy as np
from network_model import (
    build_platform_network,
    inject_disruption,
    cascading_failure_step,
    network_health,
    reset_network,
)


ACTIONS = [
    "recommend_normally",       # baseline recommendation strategy
    "boost_trusted_sellers",    # favor high-trust sellers -> lowers risk, may lower revenue
    "apply_discount",           # short-term boost to transactions/utility, costs money
    "restrict_suspicious_node", # actively isolate a low-trust node -> contains cascade
    "increase_personalization", # more data used -> higher utility, higher privacy cost
]


class DigitalPlatformEnv:
    def __init__(self, n_users=40, n_sellers=8, seed=None,
                 alpha=1.0, beta=1.0, gamma=0.5, delta=1.5, lam=0.8,
                 disruption_prob=0.15):
        self.seed = seed
        self.alpha, self.beta, self.gamma, self.delta, self.lam = alpha, beta, gamma, delta, lam
        self.disruption_prob = disruption_prob
        self.n_users, self.n_sellers = n_users, n_sellers
        self.rng = np.random.default_rng(seed)
        self.G = build_platform_network(n_users, n_sellers, seed=seed)
        self.t = 0
        self.max_steps = 40

    def reset(self):
        reset_network(self.G)
        self.t = 0
        return self._get_state()

    def _get_state(self):
        """
        Discretized state: (health_bucket, active_seller_bucket, disruption_flag)
        Health and active fraction are binned into 5 levels each -> small tabular
        state space, disruption_flag in {0,1}.
        """
        health = network_health(self.G)
        seller_nodes = [n for n, d in self.G.nodes(data=True) if d["type"] == "seller"]
        active_sellers = sum(1 for n in seller_nodes if self.G.nodes[n]["active"])
        active_fraction = active_sellers / max(1, len(seller_nodes))

        health_bucket = min(4, int(health * 5))
        active_bucket = min(4, int(active_fraction * 5))
        disruption_flag = 1 if health < 0.6 else 0

        return (health_bucket, active_bucket, disruption_flag)

    def step(self, action_idx):
        action = ACTIONS[action_idx]
        self.t += 1

        # random chance of a new disruption event each step
        if self.rng.random() < self.disruption_prob:
            inject_disruption(self.G, severity=self.rng.uniform(0.4, 0.9))

        # let trust loss propagate one step before the agent's action takes effect
        cascading_failure_step(self.G)

        # -- compute the four/five reward components --
        health_before = network_health(self.G)

        utility = 0.5          # baseline user utility
        throughput = 0.5       # baseline economic throughput (transactions)
        comp_cost = 0.1        # baseline computation cost
        privacy_penalty = 0.0
        systemic_risk = 1.0 - health_before

        if action == "recommend_normally":
            pass  # baseline, no adjustment

        elif action == "boost_trusted_sellers":
            systemic_risk *= 0.6      # lowers risk by routing users to reliable sellers
            throughput *= 0.85        # slightly less revenue (fewer novel matches)
            comp_cost += 0.05

        elif action == "apply_discount":
            utility *= 1.3            # users happier short-term
            throughput *= 1.2         # more transactions
            comp_cost += 0.15         # cost of the discount itself

        elif action == "restrict_suspicious_node":
            # actively isolate the least-trusted node to contain cascade
            worst_node = min(self.G.nodes, key=lambda n: self.G.nodes[n]["trust"])
            self.G.nodes[worst_node]["active"] = False
            systemic_risk *= 0.4
            throughput *= 0.9          # losing one node's transactions
            comp_cost += 0.2           # cost of monitoring/enforcement

        elif action == "increase_personalization":
            utility *= 1.25
            throughput *= 1.1
            privacy_penalty = 0.6      # explicit privacy cost
            comp_cost += 0.1

        health_after = network_health(self.G)
        # reward the agent extra if health improved after its action
        systemic_risk *= (1.2 - health_after)

        reward = (
            self.alpha * utility
            + self.beta * throughput
            - self.gamma * comp_cost
            - self.delta * systemic_risk
            - self.lam * privacy_penalty
        )

        done = self.t >= self.max_steps
        info = {
            "action": action,
            "health": health_after,
            "utility": utility,
            "throughput": throughput,
            "comp_cost": comp_cost,
            "systemic_risk": systemic_risk,
            "privacy_penalty": privacy_penalty,
        }
        return self._get_state(), reward, done, info
