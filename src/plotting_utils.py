"""Shared plotting utilities for ME595 notebooks.

Lives in src/ at the project root. Notebooks add src/ to sys.path at runtime:
    sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Phase portraits ───────────────────────────────────────────────────────────

_DEFAULT_LABELS = ["x", "θ₁", "θ₂", "ẋ", "θ̇₁", "θ̇₂"]
_UNITS = ["(m)", "(rad)", "(rad)", "(m/s)", "(rad/s)", "(rad/s)"]

_PORTRAIT_PAIRS = [(0, 3), (1, 2), (1, 4), (2, 5), (0, 1), (3, 4)]


def plot_phase_portraits(
    X: np.ndarray,
    state_labels: list = _DEFAULT_LABELS,
    title: str = "",
    overlay: np.ndarray | None = None,
    overlay_label: str = "trajectory",
    overlay_color: str = "crimson",
) -> None:
    """2×3 phase-portrait scatter plots of a state dataset.

    X       : (N, 6) state dataset
    overlay : optional (M, 6) trajectory drawn on top — e.g., a failing rollout
    """
    labels_with_units = [f"{l} {u}" for l, u in zip(state_labels, _UNITS)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (i, j) in zip(axes.flat, _PORTRAIT_PAIRS):
        ax.scatter(X[:, i], X[:, j], s=1, alpha=0.25, rasterized=True, label="data")
        if overlay is not None:
            ax.plot(overlay[:, i], overlay[:, j], color=overlay_color, lw=1.5,
                    label=overlay_label, zorder=5)
            ax.scatter(overlay[0, i], overlay[0, j], color=overlay_color,
                       s=60, marker="o", zorder=6)
            ax.scatter(overlay[-1, i], overlay[-1, j], color=overlay_color,
                       s=80, marker="X", zorder=6)
        ax.set_xlabel(labels_with_units[i])
        ax.set_ylabel(labels_with_units[j])

    if overlay is not None:
        axes.flat[0].legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


# ── Rollout trajectory ────────────────────────────────────────────────────────

def plot_rollout_trajectory(
    states_6: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dt: float,
    title: str = "",
    max_steps: int = 1000,
) -> None:
    """2×3 state trajectory plot matching full-order-simulation.ipynb style.

    states_6 : (N, 6) — [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
    actions  : (N,)   — cart force
    rewards  : (N,)   — per-step reward
    """
    n     = len(actions)
    time  = np.arange(n) * dt

    x_arr   = states_6[:, 0]
    th1_deg = np.degrees(states_6[:, 1])
    th2_deg = np.degrees(states_6[:, 2])
    xdot    = states_6[:, 3]
    th1dot  = states_6[:, 4]
    th2dot  = states_6[:, 5]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    (ax_x, ax_th, ax_act), (ax_xd, ax_thd, ax_rew) = axes

    ax_x.plot(time, x_arr, color="steelblue", lw=1.2, label="x_cart")
    ax_x.axhline( 0.20, color="red", ls="--", lw=1, label="±limit")
    ax_x.axhline(-0.20, color="red", ls="--", lw=1)
    ax_x.set_xlabel("Time (s)"); ax_x.set_ylabel("Position (m)")
    ax_x.set_title("Cart position"); ax_x.legend(fontsize=8)

    ax_th.plot(time, th1_deg, color="darkorange", lw=1.2, label="θ₁")
    ax_th.plot(time, th2_deg, color="mediumpurple", lw=1.2, label="θ₂")
    ax_th.axhline(0, color="gray", ls="--", lw=1)
    ax_th.set_xlabel("Time (s)"); ax_th.set_ylabel("Angle (deg)")
    ax_th.set_title("Pole angles (0 = upright)"); ax_th.legend(fontsize=8)

    ax_act.plot(time, actions, color="teal", lw=1.0, alpha=0.8)
    ax_act.axhline( 1, color="gray", ls=":", lw=1)
    ax_act.axhline(-1, color="gray", ls=":", lw=1)
    ax_act.set_xlabel("Time (s)"); ax_act.set_ylabel("Force (normalised)")
    ax_act.set_title("Cart force action")

    ax_xd.plot(time, xdot, color="steelblue", lw=1.0, alpha=0.8)
    ax_xd.axhline(0, color="gray", ls="--", lw=1)
    ax_xd.set_xlabel("Time (s)"); ax_xd.set_ylabel("Velocity (m/s)")
    ax_xd.set_title("Cart velocity")

    ax_thd.plot(time, th1dot, color="darkorange", lw=1.0, alpha=0.8, label="θ̇₁")
    ax_thd.plot(time, th2dot, color="mediumpurple", lw=1.0, alpha=0.8, label="θ̇₂")
    ax_thd.axhline(0, color="gray", ls="--", lw=1)
    ax_thd.set_xlabel("Time (s)"); ax_thd.set_ylabel("Angular vel. (rad/s)")
    ax_thd.set_title("Pole angular velocities"); ax_thd.legend(fontsize=8)

    ax_rew.plot(time, rewards, color="coral", lw=1.0, alpha=0.8)
    ax_rew.axhline(np.mean(rewards), color="red", ls="--", lw=1.5,
                   label=f"mean = {np.mean(rewards):.3f}")
    ax_rew.set_xlabel("Time (s)"); ax_rew.set_ylabel("Step reward")
    ax_rew.set_title("Per-step reward"); ax_rew.legend(fontsize=8)

    status = "TASK COMPLETE" if n >= max_steps else f"FAILED at step {n}"
    color  = "darkgreen" if n >= max_steps else "red"
    full   = f"{title}  |  {n}/{max_steps} steps = {n*dt:.0f} s  ({status})"
    fig.suptitle(full.strip("  |  "), fontsize=11, fontweight="bold", color=color)
    plt.tight_layout()
    plt.show()


# ── Training results (learning curves + final eval) ──────────────────────────

def plot_training_results(
    ep_lengths: list,
    ep_rewards: list,
    eval_steps: np.ndarray,
    eval_lengths: np.ndarray,
    eval_rewards: np.ndarray,
    total_steps: int,
    max_steps: int,
    dt: float,
    title: str = "",
) -> None:
    """2×2 figure: learning curves (length + reward) and final eval bars.

    ep_lengths  : per-episode lengths from the final evaluation run
    ep_rewards  : per-episode rewards from the final evaluation run
    eval_steps  : (M,) timestep checkpoints from EvalCallback (already /1e6 if desired)
    eval_lengths: (M,) mean episode lengths at each checkpoint
    eval_rewards: (M,) mean rewards at each checkpoint
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    (ax_len, ax_rew), (ax_ep_len, ax_ep_rew) = axes

    breakthrough = eval_steps[eval_lengths >= max_steps]

    ax_len.axhspan(max_steps, max_steps * 1.05, color="green", alpha=0.15, zorder=0,
                   label=f"Task complete  ({max_steps} steps = {max_steps * dt:.0f} s)")
    ax_len.plot(eval_steps, eval_lengths, color="steelblue", lw=1.5)
    ax_len.fill_between(eval_steps, eval_lengths, alpha=0.15, color="steelblue")
    ax_len.axhline(max_steps, color="green", ls="--", lw=1.5)
    if len(breakthrough):
        ax_len.axvline(breakthrough[0], color="red", ls=":", lw=1,
                       label=f"first solve @ {breakthrough[0]:.2f}M steps")
    ax_len.set_ylim(0, max_steps * 1.04)
    ax_len.set_xlabel("Training steps (M)"); ax_len.set_ylabel("Mean episode length")
    ax_len.set_title("Learning curve — episode length"); ax_len.legend(fontsize=8)

    ax_rew.plot(eval_steps, eval_rewards, color="coral", lw=1.5)
    ax_rew.fill_between(eval_steps, eval_rewards, alpha=0.15, color="coral")
    ax_rew.axhline(0, color="gray", ls="--", lw=1, label="reward = 0")
    if len(breakthrough):
        ax_rew.axvline(breakthrough[0], color="red", ls=":", lw=1)
    ax_rew.set_xlabel("Training steps (M)"); ax_rew.set_ylabel("Mean cumulative reward")
    ax_rew.set_title("Learning curve — reward"); ax_rew.legend(fontsize=8)

    eps = range(1, len(ep_lengths) + 1)
    ax_ep_len.axhspan(max_steps, max_steps * 1.05, color="green", alpha=0.15, zorder=0,
                      label=f"Task complete  ({max_steps} steps)")
    ax_ep_len.bar(eps, ep_lengths, color="steelblue", zorder=2)
    ax_ep_len.axhline(np.mean(ep_lengths), color="red", ls="--", zorder=3,
                      label=f"mean = {np.mean(ep_lengths):.0f}")
    ax_ep_len.axhline(max_steps, color="green", ls="--", lw=1.5, zorder=3)
    ax_ep_len.set_ylim(0, max_steps * 1.04)
    ax_ep_len.set_xlabel("Episode"); ax_ep_len.set_ylabel("Steps")
    ax_ep_len.set_title("Final eval — episode length"); ax_ep_len.legend(fontsize=8)

    ax_ep_rew.bar(eps, ep_rewards, color="coral", zorder=2)
    ax_ep_rew.axhline(np.mean(ep_rewards), color="red", ls="--", zorder=3,
                      label=f"mean = {np.mean(ep_rewards):.1f}")
    ax_ep_rew.axhline(0, color="gray", ls=":", alpha=0.7, zorder=3)
    ax_ep_rew.set_xlabel("Episode"); ax_ep_rew.set_ylabel("Cumulative reward")
    ax_ep_rew.set_title("Final eval — episode reward"); ax_ep_rew.legend(fontsize=8)

    suptitle = title or f"PPO training — {total_steps / 1e6:.1f}M steps"
    fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.show()


# ── Evaluation comparison bars ────────────────────────────────────────────────

_PALETTE_LEN = ["steelblue", "darkorange", "mediumpurple", "seagreen"]
_PALETTE_REW = ["coral", "sandybrown", "plum", "mediumaquamarine"]


def plot_eval_bars(
    results: dict,
    max_steps: int,
    dt: float,
    title: str = "",
) -> None:
    """Comparison bar chart for multiple policy conditions.

    results : {"label": (ep_lengths_array, ep_rewards_array), ...}

    Style matches full-order-simulation.ipynb: steelblue/coral palette,
    red mean line, green max-steps line, individual episode dots overlaid.
    """
    labels = list(results.keys())
    n      = len(labels)

    means_len = [np.mean(results[l][0]) for l in labels]
    stds_len  = [np.std(results[l][0])  for l in labels]
    means_rew = [np.mean(results[l][1]) for l in labels]
    stds_rew  = [np.std(results[l][1])  for l in labels]

    x = np.arange(n)

    fig, (ax_len, ax_rew) = plt.subplots(1, 2, figsize=(13, 5))

    # Episode length
    ax_len.bar(x, means_len, 0.5,
               yerr=stds_len, capsize=5, error_kw={"lw": 1.5, "capthick": 1.5},
               color=_PALETTE_LEN[:n], zorder=2)
    for i, label in enumerate(labels):
        ep_lens = np.asarray(results[label][0])
        ax_len.scatter(np.full(len(ep_lens), x[i]), ep_lens,
                       color="black", s=12, alpha=0.4, zorder=3)
    ax_len.axhspan(max_steps, max_steps * 1.05, color="green", alpha=0.15, zorder=0)
    ax_len.axhline(max_steps, color="green", ls="--", lw=1.5,
                   label=f"max ({max_steps} steps)")
    ax_len.set_xticks(x); ax_len.set_xticklabels(labels)
    ax_len.set_ylim(0, max_steps * 1.08)
    ax_len.set_ylabel("Episode length (steps)")
    ax_len.set_title("Episode length"); ax_len.legend(fontsize=8)

    # Episode reward
    ax_rew.bar(x, means_rew, 0.5,
               yerr=stds_rew, capsize=5, error_kw={"lw": 1.5, "capthick": 1.5},
               color=_PALETTE_REW[:n], zorder=2)
    for i, label in enumerate(labels):
        ep_rews = np.asarray(results[label][1])
        ax_rew.scatter(np.full(len(ep_rews), x[i]), ep_rews,
                       color="black", s=12, alpha=0.4, zorder=3)
    ax_rew.axhline(0, color="gray", ls=":", alpha=0.7, zorder=1)
    ax_rew.set_xticks(x); ax_rew.set_xticklabels(labels)
    ax_rew.set_ylabel("Cumulative episode reward")
    ax_rew.set_title("Episode reward")

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


# ── Neural-network architecture diagram ──────────────────────────────────────

_NET_COLORS = {"input": "#4682B4", "hidden": "#E8871E", "output": "#2E8B57"}
_X_STEP     = 2.5
_NODE_R     = 0.13
_Y_RANGE    = 2.2
_MAX_SHOW   = 9
_HALF_SHOWN = 4


def _layer_node_ys(n: int):
    if n == 1:
        return np.array([0.0]), False
    if n <= _MAX_SHOW:
        return np.linspace(-_Y_RANGE, _Y_RANGE, n), False
    top = np.linspace(_Y_RANGE, 0.55, _HALF_SHOWN)
    bot = np.linspace(-0.55, -_Y_RANGE, _HALF_SHOWN)
    return np.concatenate([top, bot]), True


def plot_network_diagram(
    arch: list,
    layer_names: list | None = None,
    input_labels: list | None = None,
    output_label: str = r"$u$",
    title: str = "",
) -> None:
    """Matplotlib diagram of a fully-connected network.

    arch         : layer widths, e.g. [9, 64, 64, 1]
    layer_names  : display label under each column (auto-generated if None)
    input_labels : label for each input node (auto-generated if None)
    output_label : label for the single output node
    title        : figure title
    """
    n = len(arch)

    if layer_names is None:
        layer_names = []
        for i, w in enumerate(arch):
            if i == 0:
                layer_names.append(f"Input\n({w}-dim)")
            elif i == n - 1:
                layer_names.append("Output")
            else:
                layer_names.append("Hidden")

    if input_labels is None:
        input_labels = [f"$x_{{{i}}}$" for i in range(arch[0])]

    colors = [
        _NET_COLORS["input"] if i == 0
        else _NET_COLORS["output"] if i == n - 1
        else _NET_COLORS["hidden"]
        for i in range(n)
    ]
    x_pos  = [i * _X_STEP for i in range(n)]
    all_ys = [_layer_node_ys(w) for w in arch]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-2.6, x_pos[-1] + 1.3)
    ax.set_ylim(-_Y_RANGE - 1.7, _Y_RANGE + 1.0)
    ax.axis("off")

    for i in range(n - 1):
        for y0 in all_ys[i][0]:
            for y1 in all_ys[i + 1][0]:
                ax.plot([x_pos[i], x_pos[i + 1]], [y0, y1],
                        color="#DEDEDE", lw=0.3, zorder=1)

    for i, (x, (ys, ellipsis), color, name) in enumerate(
            zip(x_pos, all_ys, colors, layer_names)):
        for y in ys:
            ax.add_patch(plt.Circle((x, y), _NODE_R, color=color,
                                    zorder=3, ec="white", lw=1.5))
        if ellipsis:
            ax.add_patch(plt.Circle((x, 0), _NODE_R, color="white",
                                    zorder=3, ec="white", lw=1.5))
            ax.text(x, 0, r"$\vdots$", ha="center", va="center",
                    fontsize=14, color=color, zorder=4)
        if i == 0:
            for y, lbl in zip(ys, input_labels):
                ax.text(x - _NODE_R - 0.1, y, lbl,
                        ha="right", va="center", fontsize=9)
        if i == n - 1:
            ax.text(x + _NODE_R + 0.12, ys[0], output_label,
                    ha="left", va="center", fontsize=10, fontweight="bold")
        ax.text(x, -_Y_RANGE - 0.55, name,
                ha="center", va="top", fontsize=9, fontweight="bold", color=color)
        ax.text(x, -_Y_RANGE - 1.05, f"n = {arch[i]}",
                ha="center", va="top", fontsize=8, color="gray")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.show()
