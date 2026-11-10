"""
train.py
Task 6 & 7: Training loop for the DQN agent in MetaDrive.

Usage:
    python train.py --episodes 20      # quick debug run
    python train.py --episodes 800     # full training run
"""

import argparse
import json
import os
import time

import numpy as np

from make_env import make_metadrive_env
from env_utils import flatten_obs, discrete_to_continuous_action, ACTION_SIZE
from dqn_agent import DQNAgent


def train(
    num_episodes: int = 20,
    max_steps_per_episode: int = 1000,
    learning_starts: int = 1000,   # min transitions in buffer before learning
    save_path: str = "models/dqn_trained.pt",
    log_path: str = "logs/training_log.json",
    seed: int = 0,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = make_metadrive_env(start_seed=seed, use_render=False, horizon=max_steps_per_episode)

    # Determine state size from one reset
    obs, info = env.reset()
    state = flatten_obs(obs)
    state_size = state.shape[0]

    agent = DQNAgent(state_size=state_size, action_size=ACTION_SIZE)

    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_crashed": [],
        "episode_losses": [],
    }

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = flatten_obs(obs)

        episode_reward = 0.0
        episode_loss = []
        crashed = False

        for step in range(max_steps_per_episode):
            action_idx = agent.select_action(state)
            continuous_action = discrete_to_continuous_action(action_idx)

            next_obs, reward, terminated, truncated, info = env.step(continuous_action)
            next_state = flatten_obs(next_obs)
            done = terminated or truncated

            agent.store_transition(state, action_idx, reward, next_state, done)
            agent.update_epsilon()

            if len(agent.replay_buffer) >= learning_starts:
                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)

            state = next_state
            episode_reward += reward

            if info.get("crash", False) or info.get("crash_vehicle", False):
                crashed = True

            if done:
                break

        history["episode_rewards"].append(episode_reward)
        history["episode_lengths"].append(step + 1)
        history["episode_crashed"].append(bool(crashed))
        history["episode_losses"].append(float(np.mean(episode_loss)) if episode_loss else None)

        elapsed = time.time() - start_time
        print(f"[Ep {episode:4d}/{num_episodes}] "
              f"reward={episode_reward:8.2f}  len={step+1:4d}  "
              f"crash={crashed}  eps={agent.epsilon:.3f}  "
              f"buffer={len(agent.replay_buffer):6d}  time={elapsed:.1f}s")

        # periodic checkpoint
        if episode % 50 == 0:
            agent.save(save_path)
            with open(log_path, "w") as f:
                json.dump(history, f)

    # final save
    agent.save(save_path)
    with open(log_path, "w") as f:
        json.dump(history, f)

    env.close()
    print(f"Training complete. Model saved to {save_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="models/dqn_trained.pt")
    parser.add_argument("--log_path", type=str, default="logs/training_log.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        learning_starts=args.learning_starts,
        save_path=args.save_path,
        log_path=args.log_path,
        seed=args.seed,
    )
