import torch
from collections import deque
from gym import Env
import numpy as np
from gym.spaces import Discrete, Box

# Tile states
STATE_HIDDEN        = -1   # hidden/unopened 
STATE_REVEALED_MINE = -2   # mine revealed when clicked

class MinesweeperEnv(Env):
    """
    Minesweeper environment (click-only) for Deep-RL, with a legal-action mask.
    Observation: float32 image shape=(H, W, 3):
      channel 0 = hidden mask (1 if hidden, else 0)
      channel 1 = normalized revealed count (0–1)
    Action: Discrete(H * W), index = x * W + y, click only.

    reset() and step() return obs, reward, done, info
    where info["action_mask_click"] is an int8 tensor of shape (H,W)
    with 1 = legal click, 0 = illegal.
    """

    def __init__(
        self,
        size_x=9,
        size_y=9,
        mine_prob=0.15,
        deterministic=True,
        secure_first_click=False,
        device=None
    ):
        super().__init__()
        # choose device
        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else (
                    "mps" if (hasattr(torch, "mps") and torch.mps.is_available()) else "cpu"
                )
            )
        )

        self.size_x = size_x
        self.size_y = size_y
        self.mine_prob = mine_prob
        self.secure_first_click = secure_first_click
        self.deterministic = deterministic
        self.max_steps = size_x * size_y

        # action & observation spaces
        self.n_actions = size_x * size_y
        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(size_x, size_y, 2),
            dtype=np.float32
        )

        # placeholders
        self.state        = None
        self.is_mine      = None
        self.adjacent     = None
        self.action_mask_click = None
        self.win          = False

        self.reset()

    def reset(self):
        self.first_click   = True
        self.done          = False
        self.steps         = 0
        self.clicked_count = 0
        self.win           = False

        self.state = torch.full(
            (self.size_x, self.size_y),
            fill_value=STATE_HIDDEN,
            dtype=torch.int8,
            device=self.device
        )
        self.is_mine = torch.zeros(
            (self.size_x, self.size_y),
            dtype=torch.bool,
            device=self.device
        )
        self.adjacent = torch.zeros(
            (self.size_x, self.size_y),
            dtype=torch.int8,
            device=self.device
        )

        self.action_mask_click = torch.zeros(
            (self.size_x, self.size_y),
            dtype=torch.int8,
            device=self.device
        )
        self._update_mask()

        return self._get_obs(), {
            "action_mask_click": self.action_mask_click.clone(),
            "win": self.win
        }

    def step(self, action):
        x, y = divmod(int(action), self.size_y)
        self.steps += 1

        # place mines on first click
        if self.first_click:
            self._place_mines(x, y)

        # invalid click
        if self.state[x, y] != STATE_HIDDEN:
            self._update_mask()
            return self._get_obs(), -1.0, False, {
                "action_mask_click": self.action_mask_click.clone(),
                "win": self.win
            }

        # clicked a mine
        if self.is_mine[x, y]:
            self.state[x, y] = STATE_REVEALED_MINE
            self.done = True
            self._update_mask()
            return self._get_obs(), -3.0, True, {
                "action_mask_click": self.action_mask_click.clone(),
                "win": self.win
            }

        # safe reveal
        is_guess = not self._has_revealed_neighbor(x, y)
        self._flood_fill(x, y)
        reward = -1.0 if is_guess and not self.first_click else 1.0

        # check win
        total_safe = self.size_x * self.size_y - int(self.is_mine.sum().item())
        if self.clicked_count >= total_safe:
            reward += 15.0
            self.win = True
            self.done = True

        # step‐limit termination
        if not self.done and self.steps >= self.max_steps:
            self.done = True

        self.first_click = False
        self._update_mask()
        return self._get_obs(), reward, self.done, {
            "action_mask_click": self.action_mask_click.clone(),
            "win": self.win
        }

    def _has_revealed_neighbor(self, x, y) -> bool:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size_x and 0 <= ny < self.size_y:
                    if self.state[nx, ny] >= 0:
                        return True
        return False

    def _update_mask(self):
        m = self.action_mask_click
        m.fill_(0)
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.state[i, j] == STATE_HIDDEN:
                    if self.first_click or self._has_revealed_neighbor(i, j):
                        m[i, j] = 1

    def _place_mines(self, safe_x, safe_y):
        self.mine_rng = torch.Generator(device=self.device)
        if self.deterministic:
            self.mine_rng.manual_seed(42)
        else:
            self.mine_rng.seed()
        rand = torch.rand(
            (self.size_x, self.size_y),
            generator=self.mine_rng,
            device=self.device
        )
        mask = rand < self.mine_prob

        if self.secure_first_click:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = safe_x + dx, safe_y + dy
                    if 0 <= nx < self.size_x and 0 <= ny < self.size_y:
                        mask[nx, ny] = False

        self.is_mine.copy_(mask)
        adj = torch.zeros_like(self.adjacent, dtype=torch.int8, device=self.device)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                src_x0 = max(0, -dx); src_x1 = self.size_x - max(0, dx)
                src_y0 = max(0, -dy); src_y1 = self.size_y - max(0, dy)
                dst_x0 = max(0, dx);  dst_x1 = self.size_x - max(0, -dx)
                dst_y0 = max(0, dy);  dst_y1 = self.size_y - max(0, -dy)
                adj[dst_x0:dst_x1, dst_y0:dst_y1] += (
                    mask[src_x0:src_x1, src_y0:src_y1].to(torch.int8)
                )
        self.adjacent.copy_(adj)

    def _flood_fill(self, sx, sy):
        q = deque([(sx, sy)])
        while q:
            i, j = q.popleft()
            if self.state[i, j] != STATE_HIDDEN:
                continue
            cnt = int(self.adjacent[i, j].item())
            self.state[i, j] = cnt
            self.clicked_count += 1
            if cnt == 0:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < self.size_x and 0 <= nj < self.size_y:
                            q.append((ni, nj))

    def _get_obs(self):
        hid = (self.state == STATE_HIDDEN).to(torch.float32)
        cnt = torch.clamp(self.state, 0, 8).to(torch.float32) / 8.0
        return torch.stack([hid, cnt], dim=-1)

    def _reveal(self):
        return self.is_mine.to(torch.uint8).cpu().numpy()