"""
run_experiment.py
------------------
Runs the full prototype pipeline:

  1. Train a Q-learning agent on the DigitalPlatformEnv
     (Neural Computation Unit component).
  2. Plot the learning curve (reward per episode).
  3. Run a "cascading failure" stress test comparing a trained agent
     vs. a random baseline policy, and plot network health over time
     (Complexity Science and Evolution Unit component).

Outputs (saved to ./results/):
  - reward_curve.png
  - network_health_comparison.png
  - training_log.txt
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import DigitalPlatformEnv, ACTIONS
from agent import QLearningAgent
from network_model import inject_disruption

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train(n_episodes=400, seed=42):
    env = DigitalPlatformEnv(seed=seed)
    agent = QLearningAgent(n_actions=len(ACTIONS))

    episode_rewards = []
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    return env, agent, episode_rewards


def plot_learning_curve(episode_rewards):
    window = 10
    smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(7, 4))
    plt.plot(episode_rewards, alpha=0.3, label="Episode reward")
    plt.plot(range(window - 1, len(episode_rewards)), smoothed, label=f"Moving avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-learning agent: reward over training episodes")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "reward_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def run_disruption_stress_test(trained_agent, n_steps=30, seed=123):
    """
    Compares network health over time under a severe disruption event,
    for a trained agent vs. a random-action baseline policy.
    """
    def rollout(policy_fn):
        env = DigitalPlatformEnv(seed=seed)
        state = env.reset()
        # force an early severe disruption so both runs face the same stress
        inject_disruption(env.G, severity=0.9, seed=seed)
        health_trace = []
        for _ in range(n_steps):
            action = policy_fn(state)
            state, reward, done, info = env.step(action)
            health_trace.append(info["health"])
        return health_trace

    trained_policy = lambda s: int(np.argmax(trained_agent.Q[s]))
    random_policy = lambda s: np.random.randint(len(ACTIONS))

    trained_trace = rollout(trained_policy)
    random_trace = rollout(random_policy)

    plt.figure(figsize=(7, 4))
    plt.plot(trained_trace, label="Trained RL agent", linewidth=2)
    plt.plot(random_trace, label="Random baseline policy", linestyle="--")
    plt.xlabel("Step after disruption")
    plt.ylabel("Network health (0-1)")
    plt.title("Recovery from a severe disruption: trained agent vs. random policy")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "network_health_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    return trained_trace, random_trace


def summarize_policy(agent, env):
    """Prints the greedy action the agent has learned for every visited state."""
    lines = ["Learned policy (state -> preferred action):", ""]
    lines.append(f"{'state (health, active, disruption)':40s} -> action")
    for state, qvals in sorted(agent.Q.items()):
        best = ACTIONS[int(np.argmax(qvals))]
        lines.append(f"{str(state):40s} -> {best}  (Q={np.round(qvals, 2)})")
    text = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "training_log.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved: {path}")
    print()
    print(text[:800] + ("\n... (truncated, see training_log.txt)" if len(text) > 800 else ""))


if __name__ == "__main__":
    print("Training Q-learning agent on DigitalPlatformEnv ...")
    env, agent, episode_rewards = train(n_episodes=400)
    print(f"Finished training. Final 10-episode avg reward: {np.mean(episode_rewards[-10:]):.2f}")

    plot_learning_curve(episode_rewards)
    trained_trace, random_trace = run_disruption_stress_test(agent)
    summarize_policy(agent, env)

    print("\nAll results saved in ./results/")
