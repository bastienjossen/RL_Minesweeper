# visualize_numbers_dqn.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from config import Config
from env    import MinesweeperEnv, STATE_HIDDEN, STATE_REVEALED_MINE
from agent  import DQNAgent

# --------------------------------------------------------------------------- #
def visualize_dqn(model_path: str,
                  out_path : str = "dqn_board.gif",
                  fps      : int = 2,
                  cfg      : Config = Config()):
    """
    Roll out one episode with a trained (Dueling) DQN and save a board-by-board
    animation that shows what the agent clicks at every step.
    """
    # 1) device, env, agent -------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MinesweeperEnv(
        size_x=9, size_y=9,
        mine_prob=0.15,
        deterministic=True,         # always the same mine pattern
        secure_first_click=False,     # first click always safe
        device=device
    )

    agent = DQNAgent(env.observation_space,
                     env.action_space,
                     cfg, device)
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()

    # 2) one rollout, remember board states & actions ------------------------ #
    frames, actions = [], []
    obs, info = env.reset()
    frames.append(env.state.cpu().numpy().copy())
    done, step = False, 0

    while not done and step < env.max_steps:
        # ε = 0 → purely greedy for visualisation
        action = agent.select_action(obs, info["action_mask_click"], epsilon=0.0)
        actions.append(action)

        obs, _, done, info = env.step(action)
        frames.append(env.state.cpu().numpy().copy())
        step += 1

    # 3) build a Matplotlib animation --------------------------------------- #
    H, W = env.size_x, env.size_y
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks(np.arange(W + 1) - 0.5)
    ax.set_yticks(np.arange(H + 1) - 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black")
    ax.set_title("DQN agent solving Minesweeper")

    # text objects for each cell
    txt = [[ax.text(j, i, "", ha="center", va="center", fontsize=12)
            for j in range(W)] for i in range(H)]
    # red square for the clicked tile
    click_marker, = ax.plot([], [], "s", ms=40,
                            mfc="none", mec="red", lw=2)

    def update(k):
        board = frames[k]
        for i in range(H):
            for j in range(W):
                v = int(board[i, j])
                txt[i][j].set_text(
                    "#" if v == STATE_HIDDEN else
                    "M" if v == STATE_REVEALED_MINE else
                    str(v)
                )
        if k > 0:
            a = actions[k - 1]
            x, y = divmod(a, W)
            click_marker.set_data([y], [x])
        else:
            click_marker.set_data([], [])
        ax.set_xlabel(f"step {k}")
        return [click_marker] + [t for row in txt for t in row]

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(frames),
                                  interval=1000 / fps,
                                  blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    print(f"Saved GIF →  {out_path}")
    plt.close(fig)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    visualize_dqn(
        model_path="best.pth",       # path to your checkpoint
        out_path="dqn_board.gif",
        fps=2
    )
