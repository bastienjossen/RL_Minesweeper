# evaluate_dqn_wandb.py
import argparse, torch, numpy as np, pandas as pd
from tqdm import trange
import wandb
import time

from config import Config
from agent  import DQNAgent
from env    import MinesweeperEnv

def evaluate(model_path: str,
             episodes  : int,
             out_csv   : str,
             cfg       : Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MinesweeperEnv(
        size_x=9, size_y=9,
        mine_prob=0.15,
        deterministic=False,
        secure_first_click=True,
        device=device
    )
    agent = DQNAgent(env.observation_space,
                     env.action_space,
                     cfg, device)
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()

    records = []
    wins = 0
    for ep in trange(episodes, desc="Evaluating"):
        obs, info = env.reset()
        done, R, t = False, 0.0, 0
        while not done and t < env.max_steps:
            a, mask = agent.select_action(obs, info["action_mask_click"], epsilon=0.0), info["action_mask_click"]
            obs, r, done, info = env.step(a)
            R += r;  t += 1
        wins += int(info["win"])
        records.append({"episode": ep, "reward": R, "win": int(info["win"]), "steps": t})

    # save per-episode CSV
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    # compute summary
    stats = {
        "episodes":   episodes,
        "mean_reward": df["reward"].mean(),
        "std_reward":  df["reward"].std(),
        "win_rate":    wins / episodes,
        "mean_steps":  df["steps"].mean()
    }
    return stats, out_csv

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     type=str, default="best.pth")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--out_csv",  type=str, default="eval_results.csv")
    p.add_argument("--nameP", type=str, default="DQN")
    args = p.parse_args()

    # 1) init wandb
    wandb.init(
      project="minesweeper-RL-eval",
      name   = f"{args.nameP}_eval_{int(time.time())}",
      config = {"ckpt": args.ckpt, "episodes": args.episodes}
    )

    # 2) run evaluation
    stats, csv_path = evaluate(args.ckpt, args.episodes, args.out_csv, Config())

    # 3) log summary metrics
    wandb.log(stats)

    # 4) save the full CSV as a W&B artifact
    artifact = wandb.Artifact(
      name="eval_results",
      type="evaluation",
      description="Per-episode rewards, wins, and steps"
    )
    artifact.add_file(csv_path)
    wandb.log_artifact(artifact)

    # 5) print to console
    print("\n=== DQN evaluation ===")
    for k, v in stats.items():
        if k == "win_rate":
            print(f"{k:>12}: {v:.2%}")
        else:
            print(f"{k:>12}: {v:.2f}")