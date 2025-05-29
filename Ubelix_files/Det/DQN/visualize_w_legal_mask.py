# visualize_numbers_dqn_with_mask.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

from config import Config
from env    import MinesweeperEnv, STATE_HIDDEN, STATE_REVEALED_MINE
from agent  import DQNAgent

from matplotlib.colors import ListedColormap

def visualize_dqn_with_mask(model_path: str,
                            out_path : str = "dqn_board_with_mask.gif",
                            fps      : int = 1,
                            cfg      : Config = Config()):
    """
    Roll out one episode with a trained (Dueling) DQN, record the legal-action mask,
    and save a board-by-board animation shading illegal moves in gray.
    """
    # 1) device, env, agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MinesweeperEnv(
        size_x=9, size_y=9,
        mine_prob=0.15,
        deterministic=True,
        secure_first_click=False,
        device=device
    )

    agent = DQNAgent(env.observation_space,
                     env.action_space,
                     cfg, device)
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()

    # 2) rollout: record frames, masks, and actions
    frames, masks, actions = [], [], []
    obs, info = env.reset()
    frames.append(env.state.cpu().numpy().copy())
    masks.append(info["action_mask_click"].cpu().numpy().copy())

    done, step = False, 0
    while not done and step < env.max_steps:
        action = agent.select_action(obs, info["action_mask_click"], epsilon=0.0)
        actions.append(action)

        obs, _, done, info = env.step(action)
        frames.append(env.state.cpu().numpy().copy())
        masks.append(info["action_mask_click"].cpu().numpy().copy())

        step += 1

    # 3) plot setup
    H, W = env.size_x, env.size_y
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks(np.arange(W + 1) - 0.5)
    ax.set_yticks(np.arange(H + 1) - 0.5)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(color="black")
    ax.set_title("DQN agent solving Minesweeper\n(green = legal)")

    # text objects for cell values
    txt = [[ax.text(j, i, "", ha="center", va="center", fontsize=12)
            for j in range(W)] for i in range(H)]
    # red square for the click
    click_marker, = ax.plot([], [], "s", ms=40, mfc="none", mec="red", lw=2)
    
    green_cmap = ListedColormap(["none", "lime"])  # 0: transparent, 1: green
    overlay = ax.imshow(np.zeros((H, W)), cmap=green_cmap, alpha=0.4, vmin=0, vmax=1)

    def update(frame_idx):
        board = frames[frame_idx]
        mask  = masks[frame_idx]
        # shade illegal (where mask==0)
        overlay.set_data(mask)

        # update text
        for i in range(H):
            for j in range(W):
                v = int(board[i, j])
                txt[i][j].set_text(
                    "#" if v == STATE_HIDDEN else
                    "M" if v == STATE_REVEALED_MINE else
                    str(v)
                )

        # show last click
        if frame_idx > 0:
            a = actions[frame_idx - 1]
            x, y = divmod(a, W)
            click_marker.set_data([y], [x])
        else:
            click_marker.set_data([], [])

        ax.set_xlabel(f"Step {frame_idx}")
        return [overlay, click_marker] + [t for row in txt for t in row]

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=1000/fps,
        blit=True
    )

    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved GIF →  {out_path}")


if __name__ == "__main__":
    visualize_dqn_with_mask(
        model_path="best.pth",
        out_path="dqn_board_with_mask.gif",
        fps=1
    )
