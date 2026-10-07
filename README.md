# Reinforcement Learning for Resilient Digital Platforms
### A prototype for the OIST Research Internship application (Neural Computation Unit / Complexity Science and Evolution Unit)

## Overview

This is a small, transparent prototype demonstrating the core idea behind
the proposed research project: a **reinforcement-learning agent**
(Neural Computation Unit) that manages a digital marketplace modeled as a
**complex adaptive network** of users and sellers (Complexity Science and
Evolution Unit), and is evaluated on how well it preserves platform
health under disruptions (outages, trust attacks, cascading failures).

The Quantum Architecture Unit is not implemented here — at this stage the
prototype only establishes the classical RL + complex-network pipeline
that a future quantum or hybrid quantum-classical optimization component
could plug into (e.g., replacing the tabular Q-table with a
quantum-assisted optimizer, or exploring quantum approaches to the
underlying combinatorial decision problem).

## What the prototype does

1. **Builds a network** (`network_model.py`) of ~40 users and 8 sellers,
   connected by weighted edges representing transaction volume, plus a
   light social layer between users.
2. **Wraps it in an MDP** (`environment.py`): the platform's state is a
   discretized summary of network health; the agent chooses among five
   management actions (normal recommendations, boosting trusted sellers,
   discounts, restricting a suspicious node, or increasing
   personalization); the reward follows

   ```
   R = alpha * Utility + beta * Throughput - gamma * Cost - delta * SystemicRisk - lambda * PrivacyPenalty
   ```

3. **Trains a tabular Q-learning agent** (`agent.py`) with an
   epsilon-greedy policy over ~400 episodes.
4. **Stress-tests** the trained policy against a random baseline under a
   severe, injected disruption, tracking how network health evolves
   step by step.
5. **Visualizes** the network before/after a disruption, coloring nodes
   by trust level to show the cascading effect.

## Files

| File | Role |
|---|---|
| `network_model.py` | Builds the graph, injects disruptions, propagates cascading trust loss |
| `environment.py` | MDP wrapper: state, actions, reward function |
| `agent.py` | Tabular Q-learning agent |
| `run_experiment.py` | Trains the agent, plots learning curve and disruption stress test |
| `visualize_network.py` | Draws the network before/after a disruption |
| `results/` | Generated plots and training log |

## How to run

```bash
pip install networkx numpy matplotlib
python3 run_experiment.py       # trains the agent, saves reward curve + stress test plots
python3 visualize_network.py    # saves a before/after network diagram
```

## Results (example run, seed=42)

- **Learning curve** (`results/reward_curve.png`): average episode reward
  increases from roughly 20 to 35+ over training, showing the agent is
  learning a useful policy rather than acting randomly.
- **Disruption stress test** (`results/network_health_comparison.png`):
  after a severe, identical disruption is injected into both runs, the
  trained agent's network health degrades noticeably more slowly than
  under a random policy — a first, simple demonstration of *resilience*
  gained through learning.
- **Learned policy** (`results/training_log.txt`): in low-health /
  disruption states the agent consistently prefers
  `restrict_suspicious_node`, while in healthy states it shifts toward
  more profit-oriented actions — a sensible, interpretable strategy.
- **Network diagram** (`results/network_before_after.png`): visually
  shows trust loss spreading from a single attacked node to its direct
  neighbors.

## Honest limitations (by design, for now)

- The state space is small and discretized; a Deep Q-Network or
  actor-critic agent (PyTorch) would scale better to richer state
  representations and is the natural next step.
- The reward weights (alpha, beta, gamma, delta, lambda) are fixed
  by hand; a more rigorous version would fit or tune them against
  realistic platform data or a more principled utility model.
- The network model is a simplified stylized abstraction (no real
  transaction data), intended to demonstrate the pipeline rather than
  reproduce a real marketplace.
- No quantum component is implemented yet; this is intentionally left
  as the "prospective" extension described in the research proposal.

## Possible next steps

- Replace the tabular agent with a small Deep Q-Network for richer states.
- Add a second, competing agent (e.g., an attacker) for a game-theoretic
  / adversarial setting connecting to the Complexity Science Unit's
  interest in cooperation and competition.
- Formulate a small combinatorial sub-problem (e.g., seller-recommendation
  assignment) and compare a classical solver against a quantum circuit
  simulated in Qiskit, as a first concrete link to the Quantum
  Architecture Unit.
