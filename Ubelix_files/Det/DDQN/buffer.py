# buffer.py
import torch

class PrioritizedReplayBuffer:
    """
    capacity: max number of transitions
    alpha:    prioritization exponent
    device:   torch.device for storage & sampling
    """
    def __init__(self, capacity, alpha, device):
        self.capacity = capacity
        self.alpha    = alpha
        self.device   = device

        self.pos    = 0
        self.size   = 0
        self.states = [None]*capacity
        self.actions= [None]*capacity
        self.rewards= [None]*capacity
        self.next_s = [None]*capacity
        self.dones  = [None]*capacity
        self.prios  = torch.zeros(capacity, device=device)

    def append(self, state, action, reward, next_state, done, error=None):
        # state & next_state are expected as torch.Tensor on device
        p = (1e-2 if error is None
             else torch.clamp((error.abs() + 1e-6).pow(self.alpha), min=1e-2))
        idx = self.pos
        self.states[idx]  = state
        self.actions[idx] = torch.tensor([action], device=self.device)
        self.rewards[idx] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_s[idx]  = next_state
        self.dones[idx]   = torch.tensor([done], dtype=torch.float32, device=self.device)
        self.prios[idx]   = p

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta, uniform_frac = 1.0):
        assert self.size >= batch_size

        prios = self.prios[:self.size]
        probs = prios / prios.sum()

        prios = self.prios[:self.size]
        probs = prios / prios.sum()

        n_uniform = int(batch_size * uniform_frac)
        n_priority = batch_size - n_uniform
        idxs_p = torch.multinomial(probs, n_priority, replacement=True)
        idxs_u = torch.randint(0, self.size, (n_uniform,), device=self.device)
        idxs   = torch.cat([idxs_p, idxs_u], dim=0)

        min_prob   = probs.min().item()
        max_weight = (self.size * min_prob) ** (-beta)
        weights    = (self.size * probs[idxs]).pow(-beta) / max_weight

        S  = torch.stack([ self.states[i]  for i in idxs ]).to(self.device)
        A  = torch.stack([ self.actions[i] for i in idxs ]).to(self.device)
        R  = torch.stack([ self.rewards[i] for i in idxs ]).to(self.device)
        S2 = torch.stack([ self.next_s[i]  for i in idxs ]).to(self.device)
        D  = torch.stack([ self.dones[i]   for i in idxs ]).to(self.device)

        return S, A, R, S2, D, idxs, weights.to(self.device)

    def update_priorities(self, idxs, errors):
        for i, e in zip(idxs, errors):
            self.prios[i] = (e.abs() + 1e-6).pow(self.alpha)

    def __len__(self):
        return self.size
