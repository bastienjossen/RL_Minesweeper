import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from collections import defaultdict

from env import MinesweeperEnv, STATE_HIDDEN, STATE_REVEALED_MINE
from agent import QLearningAgent


def visualize_q_table(q_table_path,
                      out_path="qlearning_board.gif",
                      fps=2):
    # 1) Load Q-table
    with open(q_table_path, "rb") as f:
        Qdict = pickle.load(f)

    # 2) Setup deterministic env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MinesweeperEnv(
        size_x=9, size_y=9,
        mine_prob=0.15,
        deterministic=True,
        secure_first_click=False,
        device=device
    )

    # 3) Create agent & inject Q
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=0.1, gamma=0.99, epsilon=0.0  # greedy
    )
    agent.Q = defaultdict(lambda: np.zeros(env.n_actions, dtype=np.float32),
                          Qdict)

    # 4) Roll out one episode
    frames, masks, acts = [], [], []
    obs, info = env.reset()
    frames.append(env.state.cpu().numpy().copy())
    masks.append(info["action_mask_click"].cpu().numpy().copy())
    done, step = False, 0

    while not done and step < env.max_steps:
        state_np = obs.cpu().numpy()
        mask = info["action_mask_click"].cpu().numpy().flatten()
        action = agent.act(state_np, mask)
        acts.append(action)

        obs, _, done, info = env.step(action)
        frames.append(env.state.cpu().numpy().copy())
        masks.append(info["action_mask_click"].cpu().numpy().copy())
        step += 1

    # 5) Build the animation
    H, W = env.size_x, env.size_y
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks(np.arange(W + 1) - 0.5)
    ax.set_yticks(np.arange(H + 1) - 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='black')
    ax.set_title("Q-Learning Agent Solving Minesweeper\n(green = legal move)")

    # Prepare text and overlays
    text_objs = [[ax.text(j, i, "", ha="center", va="center", fontsize=12)
                  for j in range(W)] for i in range(H)]
    click_marker, = ax.plot([], [], marker='s', markersize=40,
                            markerfacecolor="none",
                            markeredgecolor="red", lw=2)

    green_cmap = ListedColormap(["none", "lime"])
    overlay = ax.imshow(np.zeros((H, W)), cmap=green_cmap, alpha=0.3, vmin=0, vmax=1)

    def update(frame_idx):
        st = frames[frame_idx]
        mask = masks[frame_idx].reshape(H, W)

        # Update text values
        for i in range(H):
            for j in range(W):
                v = int(st[i, j])
                if v == STATE_HIDDEN:
                    txt = "#"
                elif v == STATE_REVEALED_MINE:
                    txt = "M"
                else:
                    txt = str(v)
                text_objs[i][j].set_text(txt)

        # Update green overlay (legal moves)
        overlay.set_data(mask)

        # Show last click
        if frame_idx > 0:
            a = acts[frame_idx - 1]
            x, y = divmod(a, W)
            click_marker.set_data([y], [x])
        else:
            click_marker.set_data([], [])

        ax.set_xlabel(f"Step {frame_idx}")
        return [overlay, click_marker] + [t for row in text_objs for t in row]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 / fps, blit=True
    )

    ani.save(out_path, writer='pillow', fps=fps)
    print(f"Saved Q-learning board animation to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    visualize_q_table("q_table.pkl",
                      out_path="qlearning_board.gif",
                      fps=2)
