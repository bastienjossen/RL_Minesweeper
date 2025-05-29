from dataclasses import dataclass

@dataclass
class Config:

    epochs: int        = 50_000
    # replay
    mem_size: int      = 50_000
    batch_size: int    = 64
    warmup: int        = 1_000          # MIN before learning
    alpha: float       = 0.6             # PER exponent
    beta_start: float  = 0.4
    beta_inc: float    = 1e-5

    # optimisation
    lr_init: float     = 1e-2
    lr_gamma: float    = 0.99975         # exponential decay
    lr_min:  float     = 1e-3
    tau: float         = 0.002           # soft-update rate
    grad_clip: float   = 1.0

    # exploration
    eps_init: float    = 0.95
    eps_decay: float   = 0.99975
    eps_min:  float    = 0.01
    disc : float = 0.1

    # misc
    target_every: int  = 1_000           # soft update still every ep, but we’ll log here
    uniform_frac: float = 0.20           # ρ for mixed replay

    # architecture
    conv_units: int   = 128
    dense_units: int  = 512
