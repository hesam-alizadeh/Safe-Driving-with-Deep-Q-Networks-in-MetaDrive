"""
evaluate.py
Task 8: Evaluation and Visualization.

- Plots training curves (reward, episode length, crash rate) from logs/training_log.json
- Runs evaluation episodes with the trained (greedy) policy
- Optionally renders the trained agent in 3D and top-down view
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from make_env import make_metadrive_env
from env_utils import flatten_obs, discrete_to_continuous_action, ACTION_SIZE
from dqn_agent import DQNAgent


def plot_training_curves(log_path="logs/training_log.json", out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    with open(log_path, "r") as f:
        history = json.load(f)

    rewards = history["episode_rewards"]
    lengths = history["episode_lengths"]
    crashed = history["episode_crashed"]

    episodes = np.arange(1, len(rewards) + 1)

    def moving_avg(x, w=20):
        x = np.array(x, dtype=np.float32)
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    # --- Reward curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Episode reward")
    ma = moving_avg(rewards)
    plt.plot(episodes[-len(ma):], ma, label="Moving avg (20 ep)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training Reward over Episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_curve.png"))
    plt.close()

    # --- Episode length curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, lengths, alpha=0.3, label="Episode length")
    ma = moving_avg(lengths)
    plt.plot(episodes[-len(ma):], ma, label="Moving avg (20 ep)")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Length over Episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "length_curve.png"))
    plt.close()

    # --- Crash rate curve (rolling window) ---
    window = 20
    crash_arr = np.array(crashed, dtype=np.float32)
    if len(crash_arr) >= window:
        crash_rate = np.convolve(crash_arr, np.ones(window) / window, mode="valid")
        plt.figure(figsize=(8, 5))
        plt.plot(episodes[-len(crash_rate):], crash_rate)
        plt.xlabel("Episode")
        plt.ylabel(f"Crash rate (rolling {window}-ep)")
        plt.title("Crash Rate over Training")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "crash_rate.png"))
        plt.close()

    print(f"Saved plots to {out_dir}/")
    print(f"Overall crash rate: {np.mean(crashed):.2%}")
    print(f"Final 20-episode avg reward: {np.mean(rewards[-20:]):.2f}")


def evaluate_policy(model_path="models/dqn_trained.pt", num_episodes=10,
                     render_3d=False, render_topdown=False, max_steps=1000):
    """
    Run evaluation episodes with the trained greedy policy and report
    average reward and crash statistics.
    """
    use_render = render_3d
    env = make_metadrive_env(use_render=use_render, horizon=max_steps)

    obs, info = env.reset()
    state = flatten_obs(obs)
    state_size = state.shape[0]

    agent = DQNAgent(state_size=state_size, action_size=ACTION_SIZE)
    agent.load(model_path)

    eval_rewards, eval_crashes, eval_lengths = [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        state = flatten_obs(obs)
        ep_reward = 0.0
        crashed = False

        for step in range(max_steps):
            action_idx = agent.select_action(state, greedy=True)
            continuous_action = discrete_to_continuous_action(action_idx)
            obs, reward, terminated, truncated, info = env.step(continuous_action)
            state = flatten_obs(obs)
            ep_reward += reward

            if render_3d:
                env.render(mode="topdown" if render_topdown else None)

            if info.get("crash", False) or info.get("crash_vehicle", False):
                crashed = True
            if terminated or truncated:
                break

        eval_rewards.append(ep_reward)
        eval_crashes.append(crashed)
        eval_lengths.append(step + 1)
        print(f"[Eval Ep {ep+1}] reward={ep_reward:.2f}  len={step+1}  crash={crashed}")

    env.close()

    print("\n=== Evaluation Summary ===")
    print(f"Average reward : {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")
    print(f"Average length : {np.mean(eval_lengths):.1f}")
    print(f"Crash rate     : {np.mean(eval_crashes):.2%}")

    return {
        "rewards": eval_rewards,
        "lengths": eval_lengths,
        "crashes": eval_crashes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="logs/training_log.json")
    parser.add_argument("--model_path", type=str, default="models/dqn_trained.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render_3d", action="store_true")
    parser.add_argument("--render_topdown", action="store_true")
    args = parser.parse_args()

    plot_training_curves(log_path=args.log_path)
    evaluate_policy(
        model_path=args.model_path,
        num_episodes=args.episodes,
        render_3d=args.render_3d,
        render_topdown=args.render_topdown,
    )
