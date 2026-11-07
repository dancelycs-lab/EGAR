"""Generate publication figures from final event-aligned Paper3 results only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACRO = ROOT / "experiments" / "authority_calibration"
PEND = ROOT / "experiments" / "pendubot_transfer"
GLOBAL = ROOT / "experiments" / "global_recovery"
OUT = ROOT.parent / "figures"

COLORS = {
    "energy": "#0072B2",
    "critic": "#D55E00",
    "no_energy": "#777777",
    "nominal": "#777777",
    "adaptive": "#0072B2",
    "oracle": "#009E73",
}
STYLES = {
    "energy": ("-", "o"),
    "critic": ("--", "s"),
    "no_energy": (":", "^"),
    "nominal": ("--", "s"),
    "adaptive": ("-", "o"),
    "oracle": ("-.", "D"),
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def wrap_error(q: np.ndarray) -> np.ndarray:
    return (q - np.array([np.pi, 0.0]) + np.pi) % (2.0 * np.pi) - np.pi


def mechanical_energy(x: np.ndarray) -> np.ndarray:
    q1, q2, v1, v2 = np.moveaxis(x, -1, 0)
    m2 = 0.6255677234174437
    l1 = 0.2
    r2 = 0.25569305436052964
    inertia1 = 0.031887199591513114
    inertia2 = 0.05086984812807257
    mass1 = 0.5234602302310271
    gravity = 9.81
    coupling = m2 * l1 * r2
    m00 = inertia1 + inertia2 + m2 * l1**2 + 2.0 * coupling * np.cos(q2)
    m01 = inertia2 + coupling * np.cos(q2)
    potential1 = (mass1 * l1 + m2 * l1) * gravity
    potential2 = m2 * r2 * gravity
    kinetic = 0.5 * (m00 * v1**2 + 2.0 * m01 * v1 * v2 + inertia2 * v2**2)
    return kinetic - potential1 * np.cos(q1) - potential2 * np.cos(q1 + q2)


def empirical_recovery(ax: plt.Axes, rows: list[tuple[str, dict[str, np.ndarray]]]) -> None:
    for name, data in rows:
        arrivals = np.sort(data['arrival'][data['success']])
        time = np.r_[0., arrivals, 20.]
        recovered = np.r_[0., np.arange(1, len(arrivals)+1)/len(data['arrival']), len(arrivals)/len(data['arrival'])]
        style, marker = STYLES[name]
        ax.step(
            time,
            recovered,
            where='post',
            linestyle=style,
            color=COLORS[name],
            label={"energy": "EGAR", "critic": "Critic terminal", "no_energy": "Energy removed"}[name],
        )
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Dwell-qualified entry time (s)")
    ax.set_ylabel("Recovered fraction")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, color="0.88", linewidth=0.55)


def make_recovery_figure() -> None:
    acro = [
        ("energy", load_npz(ACRO / "aligned_core_energy.npz")),
        ("critic", load_npz(ACRO / "aligned_core_critic.npz")),
        ("no_energy", load_npz(ACRO / "aligned_core_no_energy.npz")),
    ]
    pend = [
        ("energy", load_npz(PEND / "aligned_confirm_energy.npz")),
        ("critic", load_npz(PEND / "aligned_confirm_tuned_critic.npz")),
        ("no_energy", load_npz(PEND / "aligned_confirm_no_energy.npz")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), sharey=True)
    empirical_recovery(axes[0], acro)
    empirical_recovery(axes[1], pend)
    axes[0].set_title("(a) Acrobot: elbow actuation")
    axes[1].set_title("(b) Pendubot: shoulder actuation")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.legend(loc="lower right", frameon=False, fontsize=7.5, labelspacing=.25, borderpad=.3)
    fig.tight_layout(w_pad=1)
    fig.savefig(OUT / "cross_system_recovery.pdf")
    plt.close(fig)


def make_gain_figure() -> None:
    rows = json.loads((ACRO / "aligned_gain_sweep_analysis.json").read_text())["rows"]
    gain = np.array([row["gain"] for row in rows])
    nominal_success = np.array([row["nominal_success"] for row in rows])
    adaptive_success = np.array([row["adaptive_success"] for row in rows])
    nominal_time = np.array([row["nominal_capped_time"] for row in rows])
    adaptive_time = np.array([row["adaptive_capped_time"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45))
    for name, values in [("nominal", nominal_success), ("adaptive", adaptive_success)]:
        style, marker = STYLES[name]
        axes[0].plot(gain, values, linestyle=style, marker=marker, color=COLORS[name], label={"nominal": "Fixed nominal model", "adaptive": "Integral calibration"}[name])
    axes[0].axvspan(0.8, 1.25, color="#56B4E9", alpha=0.10, linewidth=0)
    axes[0].set_xlabel("Effective input gain")
    axes[0].set_ylabel("Recoveries / 32")
    axes[0].set_ylim(0, 33)
    axes[0].set_yticks([0, 8, 16, 24, 32])
    axes[0].set_title("(a) Recovery count")

    for name, values in [("nominal", nominal_time), ("adaptive", adaptive_time)]:
        style, marker = STYLES[name]
        axes[1].plot(gain, values, linestyle=style, marker=marker, color=COLORS[name], label={"nominal": "Fixed nominal model", "adaptive": "Integral calibration"}[name])
    axes[1].axvspan(0.8, 1.25, color="#56B4E9", alpha=0.10, linewidth=0)
    axes[1].set_xlabel("Effective input gain")
    axes[1].set_ylabel("Capped recovery time (s)")
    axes[1].set_ylim(0, 21)
    axes[1].set_title("(b) Failure-inclusive time")
    for ax in axes:
        ax.set_xticks(gain)
        ax.grid(True, color="0.88", linewidth=0.55)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "gain_sensitivity.pdf")
    plt.close(fig)


def make_response_figure() -> None:
    data = {
        "energy": load_npz(ACRO / "aligned_core_energy.npz"),
        "critic": load_npz(ACRO / "aligned_core_critic.npz"),
        "no_energy": load_npz(ACRO / "aligned_core_no_energy.npz"),
    }
    successful = data["energy"]["arrival"][data["energy"]["success"]]
    median = np.median(successful)
    eligible = np.flatnonzero(data["energy"]["success"])
    index = int(eligible[np.argmin(np.abs(data["energy"]["arrival"][eligible] - median))])
    time = .002 + np.arange(data["energy"]["trace"].shape[0]) * 0.02
    labels = {"energy": "EGAR", "critic": "Critic terminal", "no_energy": "Energy removed"}

    fig, axes = plt.subplots(4, 1, figsize=(6.8, 6.15), sharex=True)
    for name, result in data.items():
        trace = result["trace"][:, index]
        state = trace[:, :4]
        angle_norm = np.max(np.abs(wrap_error(state[:, :2])), axis=1)
        velocity_norm = np.max(np.abs(state[:, 2:]), axis=1)
        energy_error = mechanical_energy(state) - mechanical_energy(np.array([[np.pi, 0.0, 0.0, 0.0]]))[0]
        style, _ = STYLES[name]
        axes[0].plot(time, angle_norm, linestyle=style, color=COLORS[name], label=labels[name])
        axes[1].plot(time, velocity_norm, linestyle=style, color=COLORS[name])
        axes[2].plot(time, energy_error, linestyle=style, color=COLORS[name])
        axes[3].plot(time, trace[:, 4], linestyle=style, color=COLORS[name])
    axes[0].axhline(0.15, color="0.25", linewidth=0.7, linestyle=(0, (2, 2)))
    axes[1].axhline(0.6, color="0.25", linewidth=0.7, linestyle=(0, (2, 2)))
    axes[2].axhline(0.0, color="0.25", linewidth=0.7, linestyle=(0, (2, 2)))
    axes[3].axhline(0.5, color="0.25", linewidth=0.7, linestyle=(0, (2, 2)))
    axes[3].axhline(-0.5, color="0.25", linewidth=0.7, linestyle=(0, (2, 2)))
    axes[0].set_ylabel(r"$\|e_q\|_\infty$ (rad)")
    axes[1].set_ylabel(r"$\|v\|_\infty$ (rad/s)")
    axes[2].set_ylabel(r"$E-E_\star$ (J)")
    axes[3].set_ylabel(r"$u$ (N m)")
    axes[3].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", ncol=3, frameon=False)
    for ax in axes:
        ax.set_xlim(0.0, 8.0)
        ax.grid(True, color="0.90", linewidth=0.5)
    fig.subplots_adjust(hspace=0.10)
    fig.savefig(OUT / "acrobot_response.pdf")
    plt.close(fig)


def projected_ellipse(P: np.ndarray, indices: list[int], radius: float) -> np.ndarray:
    covariance = np.linalg.inv(P)[np.ix_(indices, indices)]
    values, vectors = np.linalg.eigh(covariance)
    angle = np.linspace(0.0, 2.0 * np.pi, 500)
    circle = np.vstack((np.cos(angle), np.sin(angle)))
    return radius * vectors @ np.diag(np.sqrt(values)) @ circle


def make_calibration_certificate_figure() -> None:
    """Compare sampled Lyapunov decay and certified command margin."""
    local=load_npz(ROOT/'method/local_regulation.npz')
    stored=load_npz(GLOBAL/'lqr_matrices.npz')
    cert=json.loads((ROOT/'method/interval_sampled_certificate.json').read_text())
    gain=stored['K'].ravel().astype(float)
    command=np.abs((local['states']-np.array([np.pi,0,0,0]))@gain).max(-1)
    # Round upward from the interval effective-input bound.
    command_bound=.093
    assert cert['effective_input_upper']/1.25 < command_bound
    assert command.max() < command_bound < .5
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.9))
    for i,(color,style) in enumerate([('#0072B2','-'),('#D55E00','--'),('#009E73','-.')]):
        label=rf'$\gamma={local["gains"][i]:g}$'
        axes[0].semilogy(local['time'],local['normalized_V'][i].max(1),
                         color=color,linestyle=style,label=label)
        axes[1].semilogy(local['time'],command[i],color=color,linestyle=style,label=label)
    axes[0].semilogy(local['time'],(1-.0035/2)**np.arange(len(local['time'])),
                     color='.25',ls=':',label='Analytical envelope')
    axes[1].axhline(command_bound,color='.25',ls=':',label='Certified bound')
    axes[0].set(ylabel=r'$\max V_L(e_k)/V_L(e_0)$',title='(a) Sampled-data contraction')
    assert np.all(command>0)
    axes[1].set(ylabel=r'$\max |u_k|$ (N m)',title='(b) Local command margin',
                ylim=(command.min()*.7,.12))
    for ax in axes:
        ax.set(xlabel='Time (s)',xlim=(0,2))
        ax.grid(True,color='.90',linewidth=.5)
        ax.legend(frameon=False,fontsize=8,loc='upper right' if ax==axes[1] else 'lower left')
    fig.tight_layout()
    fig.savefig(OUT/'calibration_certificate.pdf')
    plt.close(fig)


if __name__ == "__main__":
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    make_recovery_figure()
    make_gain_figure()
    make_response_figure()
    make_calibration_certificate_figure()
