"""
make_env.py
Task 1: Create and configure the MetaDriveEnv environment.
"""

from metadrive.envs.metadrive_env import MetaDriveEnv


def make_metadrive_env(
    num_scenarios: int = 20,
    start_seed: int = 0,
    use_render: bool = False,
    horizon: int = 1000,
    traffic_density: float = 0.1,
):
    """
    Build a single-agent MetaDriveEnv configured for reproducible DQN training.

    Parameters
    ----------
    num_scenarios : number of distinct procedurally generated maps to cycle through
    start_seed    : random seed controlling map generation -> reproducibility
    use_render    : whether to open the 3D render window (keep False for headless training)
    horizon       : max steps per episode before truncation
    traffic_density : density of other (non-agent) traffic vehicles

    Returns
    -------
    env : MetaDriveEnv instance
    """
    config = dict(
        num_scenarios=num_scenarios,
        start_seed=start_seed,
        use_render=use_render,
        traffic_density=traffic_density,
        horizon=horizon,
        # Make sure crash information is exposed in the info dict returned
        # by env.step(), needed to track crash rate during training.
        crash_vehicle_done=False,   # don't auto-terminate on crash so agent
                                     # can still learn recovery; we still log it
        out_of_route_done=True,
        # Keep observation as a flat state vector (not raw images)
        image_observation=False,
    )
    env = MetaDriveEnv(config)
    return env


def sanity_check_env(env):
    """
    Task 1: verify the environment can be reset and stepped through
    before starting DQN training.
    """
    obs, info = env.reset()
    print("Reset OK. Observation type:", type(obs),
          "shape:" if hasattr(obs, "shape") else "", getattr(obs, "shape", ""))

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, terminated={terminated}, "
              f"truncated={truncated}, crash={info.get('crash', 'N/A')}")

    env.close()
    print("Sanity check finished successfully.")


if __name__ == "__main__":
    test_env = make_metadrive_env(use_render=False)
    sanity_check_env(test_env)
