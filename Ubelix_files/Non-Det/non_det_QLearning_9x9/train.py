# train_qlearning.py

import time
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env import MinesweeperEnv
from agent import QLearningAgent

# ─── Hyperparameters ───────────────────────────────────────────────────────────
EPISODES      = 50_000
ALPHA         = 0.1        # learning rate
GAMMA         = 0.99       # discount factor
EPS_START     = 1.0        # initial ε
EPS_DECAY     = 0.9999     # per-episode multiplicative decay
EPS_MIN       = 0.01       # floor on ε
ROLLING_WIND  = 2_500      # window for rolling stats
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Device & env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for env: {device}")
    env = MinesweeperEnv(
        size_x=9, size_y=9,
        mine_prob=0.15,
        deterministic=False,
        secure_first_click=True,
        device=device
    )

    # 2) Agent
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPS_START
    )

    # 3) Storage
    train_rewards = []
    win_flags     = []

    start_time = time.time()
    for ep in range(1, EPISODES+1):
        obs_t, info = env.reset()
        state       = obs_t.cpu().numpy()
        done        = False
        total_r     = 0.0
        steps       = 0

        while not done and steps < env.max_steps:
            mask      = info["action_mask_click"].cpu().numpy()
            action    = agent.act(state, mask)

            next_obs_t, reward, done, info = env.step(action)
            next_state = next_obs_t.cpu().numpy()

            agent.update(state, action, reward, next_state)

            state    = next_state
            total_r += reward
            steps   += 1

        train_rewards.append(total_r)
        win_flags.append(int(info["win"]))

        # decay epsilon
        agent.epsilon = max(EPS_MIN, agent.epsilon * EPS_DECAY)

        if ep % 1000 == 0:
            print(f"Episode {ep:5d} | Reward {total_r:6.2f} | ε {agent.epsilon:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes.")

    # ─── Build DataFrame & rolling stats ─────────────────────────────────────────
    df = pd.DataFrame({
        "episode":     np.arange(1, EPISODES+1),
        "train_reward": train_rewards,
        "win":          win_flags
    })
    df["roll_mean"] = df["train_reward"].rolling(ROLLING_WIND).mean()
    df["roll_std"]  = df["train_reward"].rolling(ROLLING_WIND).std()
    df["roll_win"]  = df["win"].rolling(ROLLING_WIND).mean()

    # ─── Plot 1: Total Reward ────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df["episode"], df["train_reward"], alpha=0.3, label="Raw reward")
    ax1.plot(df["episode"], df["roll_mean"],    label=f"{ROLLING_WIND}-ep mean")
    ax1.fill_between(
        df["episode"],
        df["roll_mean"] - df["roll_std"],
        df["roll_mean"] + df["roll_std"],
        alpha=0.2
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Q-Learning: Total Reward")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig("qlearning_reward_curve.png")
    plt.show()

    # ─── Plot 2: Winrate ─────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(df["episode"], df["roll_win"], label=f"{ROLLING_WIND}-ep winrate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Winrate")
    ax2.set_title("Q-Learning: Rolling Winrate")
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig("qlearning_winrate_curve.png")
    plt.show()

    # ─── Persist results ─────────────────────────────────────────────────────────
    df.to_csv("qlearning_minesweeper_results.csv", index=False)
    print("Saved training data to qlearning_minesweeper_results.csv")

    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)
    print("Saved Q-table to q_table.pkl")

if __name__ == "__main__":
    main()
