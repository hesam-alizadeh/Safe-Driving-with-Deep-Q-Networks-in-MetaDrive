# Safe Driving with DQN in MetaDrive

A complete implementation of the assignment “Safe Driving with a Deep Q-Network in the MetaDrive Simulator.”

## Team Members
## Hesam Alizadeh

## File Structure

```text
metadrive_dqn/
├── env_utils.py        # Task 2: flatten_obs, discrete_to_continuous_action
├── make_env.py         # Task 1: Create and test the MetaDriveEnv environment
├── dqn_network.py      # Task 3: Q-network (DQNNetwork)
├── replay_buffer.py    # Task 4: Experience replay buffer (ReplayBuffer)
├── dqn_agent.py        # Task 5: DQN agent (epsilon-greedy + Bellman update)
├── train.py            # Tasks 6 & 7: Training loop
├── evaluate.py         # Task 8: Plotting, evaluation, and visualization
├── requirements.txt
├── models/             # Saved model goes here -> dqn_trained.pt
├── logs/               # Training log (training_log.json)
└── plots/              # Reward / episode length / crash rate plots
```

## 1. Installing Dependencies

Using a separate virtual environment is recommended (MetaDrive recommends Python 3.9 or 3.10):

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install metadrive-simulator
pip install torch --index-url https://download.pytorch.org/whl/cu121   # If you have a GPU
# Or the CPU version:
# pip install torch

pip install numpy matplotlib
```

Important MetaDrive installation notes:

- It works on Linux, Windows, and macOS; 3D rendering requires a GPU or at least an OpenGL driver.
- If you are working on a headless server (such as Colab or over SSH), set `use_render=False` (the default in this code) and enable it only for the final visualization stage.
- After installation, test it:

```bash
python -c "import metadrive; print(metadrive.__version__)"
```

## 2. Initial Environment Test (Task 1)

```bash
python make_env.py
```

This script resets the environment and performs a few random steps to make sure the environment works correctly.

## 3. Running a Short Training Session for Debugging (Task 7 — First Stage)

```bash
python train.py --episodes 20 --learning_starts 500
```

This run is only intended to verify that the implementation is correct (correctness matters more than speed).

## 4. Running Full Training (Task 7 — Final Stage)

```bash
python train.py --episodes 800 --max_steps 1000 --learning_starts 2000
```

- The final model is saved to `models/dqn_trained.pt`.
- The complete training log (reward, length, crash) is saved to `logs/training_log.json`.
- An intermediate checkpoint is also saved every 50 episodes.

Full training (500–1,000 episodes) may take several hours on a CPU; it will be faster if a GPU is available (PyTorch detects the GPU automatically).

## 5. Evaluation and Plotting (Task 8)

```bash
python evaluate.py --episodes 10
```

This command:

1. Creates `reward_curve.png`, `length_curve.png`, and `crash_rate.png` in the `plots/` directory.
2. Evaluates the saved model over 10 episodes using a greedy policy (without exploration), and prints the average reward and crash rate.

To view the agent graphically (in 3D or top-down view), run the following on a system with a display/GPU:

```bash
python evaluate.py --episodes 3 --render_3d
python evaluate.py --episodes 3 --render_3d --render_topdown
```

## Overall Workflow (Summary of Steps)

1. Install the dependencies as described in Section 1.
2. Run `make_env.py` to test the environment.
3. Run `train.py --episodes 20` for debugging.
4. After verifying correctness, run `train.py --episodes 800` (or more) for final training.
5. Run `evaluate.py` to generate plots and perform the final evaluation.
6. Review the plots in `plots/` and answer the analytical questions: Has the reward increased? How has episode length changed? Has the crash rate decreased?
7. Submit: source code + `plots/` + `models/dqn_trained.pt` + `logs/training_log.json` + answers to the questions.

## Important Implementation Notes

- **Discrete action space:** Six actions are defined in `env_utils.py` according to the table in the PDF (combinations of steering and acceleration/braking).
- **Bellman update:** In `dqn_agent.py`, the `learn()` function implements the exact formula `y = r + (1-d) * gamma * max_a' Q_target(s', a')` with MSE loss.
- **Target network:** Synchronized every `target_update_freq` learning steps (1,000 by default).
- **Epsilon-greedy:** Epsilon decreases linearly from 1.0 to 0.05 over `epsilon_decay_steps`.
- If observations in your MetaDrive version are dictionaries rather than simple arrays, `flatten_obs` automatically concatenates all subarrays, so no manual changes are needed.
- If you encounter an error related to `info["crash"]`, your installed version may use a different key, such as `"crash_vehicle"` or `"out_of_road"`. The code checks both cases, but it is best to run `print(info)` once to inspect the available keys.
