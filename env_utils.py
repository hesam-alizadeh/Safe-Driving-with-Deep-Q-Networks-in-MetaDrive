"""
env_utils.py
Task 2: Observation and Action processing utilities for MetaDrive + DQN.
"""

import numpy as np

# ---------------------------------------------------------------------
# Discrete action table (Task 1 description in the PDF)
# ---------------------------------------------------------------------
# index : (steering, throttle/brake)
# steering:  -1.0 = full left, 0.0 = straight, +1.0 = full right
# accel:     positive = forward/throttle, negative = brake
DISCRETE_ACTIONS = {
    0: (-0.5, -0.8),  # Left  + Brake
    1: (-0.5,  0.8),  # Left  + Forward
    2: ( 0.0, -0.8),  # Straight + Brake
    3: ( 0.0,  0.8),  # Straight + Forward
    4: ( 0.5, -0.8),  # Right + Brake
    5: ( 0.5,  0.8),  # Right + Forward
}

ACTION_SIZE = len(DISCRETE_ACTIONS)


def flatten_obs(obs):
    """
    Convert a MetaDrive observation into a single flat float32 numpy vector.

    MetaDrive's default observation for MetaDriveEnv is already a 1D numpy
    array (state-based observation), but sometimes it comes back as a dict
    (e.g. when image observation or multiple sub-observations are enabled).
    This function makes the code robust to both cases.
    """
    if isinstance(obs, dict):
        # Concatenate every sub-array in the dict, in a fixed (sorted) key
        # order so the resulting vector has a consistent layout.
        parts = []
        for key in sorted(obs.keys()):
            value = np.asarray(obs[key], dtype=np.float32).flatten()
            parts.append(value)
        flat = np.concatenate(parts, axis=0)
    else:
        flat = np.asarray(obs, dtype=np.float32).flatten()

    return flat.astype(np.float32)


def discrete_to_continuous_action(action_idx):
    """
    Map a discrete DQN action index (0-5) to the continuous
    [steering, throttle_or_brake] action expected by env.step() in
    MetaDrive.

    Parameters
    ----------
    action_idx : int

    Returns
    -------
    list[float] : [steering, throttle/brake] both in [-1, 1]
    """
    if action_idx not in DISCRETE_ACTIONS:
        raise ValueError(f"Invalid discrete action index: {action_idx}")
    steering, accel = DISCRETE_ACTIONS[action_idx]
    return [steering, accel]
