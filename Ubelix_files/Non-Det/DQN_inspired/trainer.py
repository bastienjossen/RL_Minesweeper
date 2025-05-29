import torch, time
import numpy as np
from config import Config
from buffer import PrioritizedReplayBuffer
from agent  import DQNAgent
from env    import MinesweeperEnv
import wandb
from utils  import config_to_dict

class Trainer:
    def __init__(self, cfg: Config, device):
        self.cfg   = cfg
        self.env   = MinesweeperEnv(size_x=9,size_y=9,
                            mine_prob=0.15,deterministic=False,
                            secure_first_click=True,device=device)
        self.agent = DQNAgent(self.env.observation_space,
                              self.env.action_space, cfg, device)
        self.buffer = PrioritizedReplayBuffer(cfg.mem_size, cfg.alpha, device)
        self.beta   = cfg.beta_start
        self.eps    = cfg.eps_init
        self.device = device

        self.best_win = 0.0                       # for checkpointing
        wandb.init(
            project="minesweeper-RL_rerun",
            name   = f"non_Det_DQN_discounted_run_{int(time.time())}",
            config = config_to_dict(cfg)
        )
        wandb.watch(self.agent.model, log="gradients", log_freq=1000)

    # -------------------------------------------------------------- #
    def run(self, episodes: int):
        for ep in range(1, episodes+1):
            state, info = self.env.reset()
            done, total_r = False, 0.0

            while not done:
                a = self.agent.select_action(state,
                                             info["action_mask_click"],
                                             self.eps)
                s2, r, done, info = self.env.step(a)
                self.buffer.append(state, a, r, s2, done)
                state, total_r = s2, total_r + r

                if len(self.buffer) > self.cfg.warmup:
                    batch = self.buffer.sample(self.cfg.batch_size,
                                               beta=self.beta,
                                               uniform_frac=self.cfg.uniform_frac)
                    td_err, loss, idxs = self.agent.learn(batch, beta=self.beta)
                    self.buffer.update_priorities(idxs, td_err)

            if ep % 1000 == 0:
                print(f"Episode {ep} | Return: {total_r:.2f} | Win: {info['win']:.2f}")

            # ---------- per-episode schedules ----------
            if len(self.buffer) > self.cfg.warmup:
                self.beta = min(1.0, self.beta + self.cfg.beta_inc)
                self.eps  = max(self.cfg.eps_min,  self.eps * self.cfg.eps_decay)
                self.agent.scheduler.step()                         # LR scheduler
                self.agent.soft_update()                            # τ≈0.002

            # ---------- logging ----------
            roll_win = int(info["win"])
            wandb.log({
                "episode":    ep,
                "total_reward":     total_r,
                "win":        roll_win,
                "epsilon":    self.eps,
                "beta":       self.beta,
                "lr":         self.agent.optimizer.param_groups[0]['lr']
            })

            # ---------- checkpoint ----------
            if roll_win > self.best_win:
                torch.save(self.agent.model.state_dict(), "best_5.pth")
                self.best_win = roll_win
                artifact = wandb.Artifact("best_5_model", type="model")
                artifact.add_file("best_5.pth")
                wandb.log_artifact(artifact)
                
        torch.save(self.agent.model.state_dict(), "final_5.pth")
        art = wandb.Artifact("final_5_model", type="model")
        art.add_file("final_5.pth")
        wandb.log_artifact(art)
