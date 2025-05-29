# agent.py
import random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from model import DuelingQNetwork
from config import Config


class DQNAgent:
    def __init__(self, observation_space, action_space, cfg: Config, device):
        h, w, c      = observation_space.shape
        n_actions    = action_space.n

        self.cfg = cfg
        self.device = device

        # networks
        self.model       = DuelingQNetwork(c, h, w, self.cfg.conv_units , self.cfg.dense_units, n_actions).to(self.device)
        self.target_net  = DuelingQNetwork(c, h, w, self.cfg.conv_units , self.cfg.dense_units, n_actions).to(self.device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        # optimizer & LR
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr_init)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.cfg.lr_gamma
        )

        # exploration & discount
        self.epsilon = self.cfg.eps_init
        self.discount= self.cfg.disc

    def select_action(self, state: torch.Tensor, legal_mask: torch.Tensor, epsilon) -> int:
        """
        state:       (H,W,2) float32 tensor on device
        legal_mask: (H,W)   int8 tensor on device
        """
        flat = legal_mask.view(-1)
        legal_idxs = torch.nonzero(flat, as_tuple=False).view(-1)
        # ε-greedy
        if torch.rand(1, device=self.device).item() < epsilon and legal_idxs.numel()>0:
            return int(legal_idxs[torch.randint(len(legal_idxs), (1,), device=self.device)])
        # exploitation
        with torch.no_grad():
            inp = state.permute(2,0,1).unsqueeze(0).float().to(self.device)
            q   = self.model(inp).squeeze(0)  # (n_actions,)
        if legal_idxs.numel() < q.numel():
            q[flat==0] = -1e9
        return int(q.argmax().item())
    
    def soft_update(self):
        tau = self.cfg.tau
        for tgt, src in zip(self.target_net.parameters(), self.model.parameters()):
            tgt.data.copy_(tau * src.data + (1.0 - tau) * tgt.data)


    def learn(self, batch, beta):
        S, A, R, S2, D, idxs, weights = batch  
        
        s  = S.permute(0, 3, 1, 2).float()             # (B,C,H,W)
        ns = S2.permute(0, 3, 1, 2).float()

        q_vals = self.model(s).gather(1, A.long())     # (B,1)

        with torch.no_grad():                          # Double-DQN target
            next_a  = self.model(ns).argmax(dim=1, keepdim=True)
            q_next  = self.target_net(ns).gather(1, next_a)
        q_target = torch.where(D == 1, R, R + self.discount * q_next)

        w = weights.view(-1, 1)                        # (B,1)
        loss = (w * F.mse_loss(q_vals, q_target, reduction="none")).mean()

        # Optimisation ------------------------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        # TD-error for PER update -------------------------------------------------------
        td_err = (q_target - q_vals).detach().squeeze(1)   # (B,)

        return td_err, loss.item(), idxs
