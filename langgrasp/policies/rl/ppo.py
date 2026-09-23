"""Minimal, dependency-free PPO (torch only) for the vectorised arm environments.

MLP actor-critic (2x256 tanh, separate actor and critic trunks), Gaussian policy with a state-independent
log-std, GAE (lambda 0.95), gamma 0.99, clip 0.2, 4 epochs, minibatch 2048 of a 64 x 64 rollout, Adam 3e-4,
advantage normalisation, running observation normalisation (saved with the checkpoint), entropy coef 0.0,
no value clipping, grad-norm clip 0.5. Episodes have a fixed horizon, so the horizon cut is treated as a
terminal (no bootstrap through the time limit; documented in docs/RL.md).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from langgrasp.policies.rl.envs import DomainRandomization, VecArmEnv

torch.set_num_threads(8)
HARDWARE = "NVIDIA L4 (x86 EC2 host)"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m_a, m_b = self.var * self.count, bv * bc
        self.var = (m_a + m_b + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, d: dict):
        self.mean, self.var, self.count = np.array(d["mean"], dtype=np.float64), np.array(d["var"], dtype=np.float64), float(d["count"])


def _mlp(inp: int, out: int, hidden: int = 256, out_gain: float = 1.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = inp
    for _ in range(2):
        lin = nn.Linear(last, hidden)
        nn.init.orthogonal_(lin.weight, gain=math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        last = hidden
    head = nn.Linear(last, out)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, log_std_init: float = -0.5):
        super().__init__()
        self.actor = _mlp(obs_dim, act_dim, hidden, out_gain=0.01)
        self.critic = _mlp(obs_dim, 1, hidden, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.actor(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


@dataclass
class PPOConfig:
    task: str = "reach"
    dr: bool = False
    n_envs: int = 64
    n_steps: int = 64
    total_steps: int = 1_500_000
    max_minutes: float = 25.0
    seed: int = 0
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 2048
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    log_std_init: float = -0.5
    device: str = "auto"
    n_threads: int = 8
    curve_every: int = 1  # log a curve point every k iterations
    init_ckpt: str = ""  # warm start: load actor-critic weights and obs normaliser from this checkpoint
    dr_ramp_frac: float = 0.0  # > 0: scale the training randomisation from 0 to 1 over this fraction of total_steps
    extra: dict = field(default_factory=dict)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(path: str, model: ActorCritic, obs_rms: RunningMeanStd, cfg: PPOConfig, dr: DomainRandomization, meta: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "actor_critic": {k: v.cpu() for k, v in model.state_dict().items()},
            "obs_rms": obs_rms.state_dict(),
            "config": asdict(cfg),
            "dr": dr.to_dict(),
            "obs_dim": model.actor[0].in_features,
            "act_dim": model.log_std.numel(),
            "hidden": model.actor[0].out_features,
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[ActorCritic, RunningMeanStd, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic(ck["obs_dim"], ck["act_dim"], ck["hidden"])
    model.load_state_dict(ck["actor_critic"])
    model.to(device).eval()
    rms = RunningMeanStd((ck["obs_dim"],))
    rms.load_state_dict(ck["obs_rms"])
    return model, rms, ck


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """rewards, values, dones: (T, N); last_value: (N,). done[t] = 1 means the transition at t ended the episode."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    gae = np.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + values


def ppo_update(
    model: ActorCritic, opt: torch.optim.Optimizer, batch: dict[str, torch.Tensor], cfg: PPOConfig, gen: torch.Generator | None = None
) -> dict[str, float]:
    """One PPO update over a flattened batch (obs, act, logp, adv, ret). Returns mean losses and stats."""
    n = batch["obs"].shape[0]
    stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "approx_kl": [], "clipfrac": []}
    g = gen or torch.Generator(device="cpu").manual_seed(cfg.seed)
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, cfg.minibatch):
            idx = perm[start : start + cfg.minibatch].to(batch["obs"].device)
            obs, act, old_logp, adv, ret = (batch[k][idx] for k in ("obs", "act", "logp", "adv", "ret"))
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            d = model.dist(obs)
            logp = d.log_prob(act).sum(-1)
            ratio = (logp - old_logp).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            v_loss = 0.5 * (model.value(obs) - ret).pow(2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pg + cfg.value_coef * v_loss - cfg.entropy_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["policy_loss"].append(float(pg))
                stats["value_loss"].append(float(v_loss))
                stats["approx_kl"].append(float((old_logp - logp).mean()))
                stats["clipfrac"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def train(cfg: PPOConfig, results_path: str | None = None, ckpt_path: str | None = None, env: VecArmEnv | None = None, verbose: bool = True) -> dict:
    """Run PPO; returns the results dict (also written to results_path as JSON if given)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    dr = DomainRandomization.train() if cfg.dr else DomainRandomization.none()
    if cfg.dr and cfg.dr_ramp_frac > 0:
        dr = DomainRandomization.scaled(0.0)
    env = env or VecArmEnv(cfg.n_envs, cfg.task, seed=cfg.seed, dr=dr, n_threads=cfg.n_threads)
    model = ActorCritic(env.obs_dim, env.act_dim, cfg.hidden, cfg.log_std_init).to(device)
    rms = RunningMeanStd((env.obs_dim,))
    if cfg.init_ckpt:
        m0, rms0, _ = load_checkpoint(cfg.init_ckpt, device)
        model.load_state_dict(m0.state_dict())
        rms.load_state_dict(rms0.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    N, T = env.n, cfg.n_steps
    obs_buf = np.zeros((T, N, env.obs_dim), dtype=np.float32)
    act_buf = np.zeros((T, N, env.act_dim), dtype=np.float32)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)
    val_buf = np.zeros((T, N), dtype=np.float32)

    raw_obs = env.reset()
    rms.update(raw_obs)
    curve: list[dict] = []
    total = 0
    it = 0
    t0 = time.time()
    env_time = 0.0
    ep_returns: list[float] = []
    ep_success: list[bool] = []
    ep_hold: list[bool] = []
    while total < cfg.total_steps and (time.time() - t0) / 60.0 < cfg.max_minutes:
        if cfg.dr and cfg.dr_ramp_frac > 0:
            env.dr = DomainRandomization.scaled(total / (cfg.dr_ramp_frac * cfg.total_steps))
        it_returns: list[float] = []
        it_success: list[bool] = []
        it_hold: list[bool] = []
        for t in range(T):
            obs_n = rms.normalize(raw_obs)
            a, logp, v = model.act(torch.as_tensor(obs_n, device=device))
            a_np = a.cpu().numpy()
            te = time.time()
            next_raw, r, d, info = env.step(a_np)
            env_time += time.time() - te
            obs_buf[t], act_buf[t], logp_buf[t], rew_buf[t], done_buf[t], val_buf[t] = obs_n, a_np, logp.cpu().numpy(), r, d, v.cpu().numpy()
            if d.any():
                it_returns += info["episode_return"][d].tolist()
                it_success += info["success"][d].tolist()
                if "hold" in info:
                    it_hold += info["hold"][d].tolist()
            raw_obs = next_raw
        total += T * N
        it += 1
        rms.update(obs_buf.reshape(-1, env.obs_dim))
        with torch.no_grad():
            last_v = model.value(torch.as_tensor(rms.normalize(raw_obs), device=device)).cpu().numpy()
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_v, cfg.gamma, cfg.lam)
        batch = {
            "obs": torch.as_tensor(obs_buf.reshape(-1, env.obs_dim), device=device),
            "act": torch.as_tensor(act_buf.reshape(-1, env.act_dim), device=device),
            "logp": torch.as_tensor(logp_buf.reshape(-1), device=device),
            "adv": torch.as_tensor(adv.reshape(-1), device=device),
            "ret": torch.as_tensor(ret.reshape(-1), device=device),
        }
        st = ppo_update(model, opt, batch, cfg, gen)
        ep_returns += it_returns
        ep_success += it_success
        ep_hold += it_hold
        if it % cfg.curve_every == 0 and it_returns:
            point = {
                "steps": total,
                "mean_return": float(np.mean(it_returns)),
                "success_rate": float(np.mean(it_success)),
                "n_episodes": len(it_returns),
                "minutes": (time.time() - t0) / 60.0,
                "std": float(model.log_std.exp().mean()),
                "approx_kl": st["approx_kl"],
            }
            if it_hold:
                point["hold_rate"] = float(np.mean(it_hold))
            curve.append(point)
            if verbose:
                print(
                    f"[{cfg.task} {'dr' if cfg.dr else 'nodr'}] it {it:4d} steps {total:8d} ret {point['mean_return']:8.3f} "
                    f"succ {point['success_rate']:.3f} std {point['std']:.3f} kl {st['approx_kl']:.4f} "
                    f"{total / (time.time() - t0):.0f} steps/s ({total / max(env_time, 1e-9):.0f} env-only) {point['minutes']:.1f} min",
                    flush=True,
                )
    minutes = (time.time() - t0) / 60.0
    tail = curve[-5:]
    result = {
        "task": cfg.task,
        "dr": cfg.dr,
        "dr_spec": dr.to_dict(),
        "n_envs": N,
        "n_steps_per_env": T,
        "total_steps": total,
        "iterations": it,
        "minutes": minutes,
        "curve": [(p["steps"], p["mean_return"], p["success_rate"]) for p in curve],
        "curve_detail": curve,
        "final_success_train_env": float(np.mean([p["success_rate"] for p in tail])) if tail else None,
        "final_success_nominal": None,
        "final_success_nominal_note": "filled by scripts/eval_ppo.py (200 deterministic episodes, no DR)",
        "throughput_steps_per_s": total / max(minutes * 60.0, 1e-9),
        "env_only_steps_per_s": total / max(env_time, 1e-9),
        "device": str(device),
        "hardware": HARDWARE,
        "config": asdict(cfg),
        "stopped_by": "steps" if total >= cfg.total_steps else "time_box",
    }
    if ckpt_path:
        save_checkpoint(ckpt_path, model, rms, cfg, dr, meta={"total_steps": total, "minutes": minutes, "hardware": HARDWARE})
        result["checkpoint"] = ckpt_path
    if results_path:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(result, f, indent=1)
    result["_model"] = model
    result["_rms"] = rms
    return result


@torch.no_grad()
def evaluate(model: ActorCritic, rms: RunningMeanStd, env: VecArmEnv, n_episodes: int = 200, deterministic: bool = True, device: str | torch.device = "cpu") -> dict:
    """Run complete episodes (fixed horizon, so all envs finish together) and return success stats with Wilson CIs."""
    model.eval()
    successes: list[bool] = []
    holds: list[bool] = []
    returns: list[float] = []
    raw = env.reset()
    while len(successes) < n_episodes:
        for _ in range(env.horizon):
            a, _, _ = model.act(torch.as_tensor(rms.normalize(raw), device=device), deterministic=deterministic)
            raw, r, d, info = env.step(a.cpu().numpy())
        assert d.all()
        need = n_episodes - len(successes)
        successes += info["success"][:need].tolist()
        returns += info["episode_return"][:need].tolist()
        if "hold" in info:
            holds += info["hold"][:need].tolist()
    k, n = int(sum(successes)), len(successes)
    lo, hi = wilson(k, n)
    out = {"n": n, "k": k, "success": k / n, "ci95": [lo, hi], "mean_return": float(np.mean(returns))}
    if holds:
        kh = int(sum(holds))
        out["hold"] = kh / n
        out["hold_ci95"] = list(wilson(kh, n))
    return out
