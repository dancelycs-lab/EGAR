"""Vector mechanical schematic of the two simulated actuator configurations.

The pose and recovery arrow are illustrative, not recorded trajectories.
Joint coordinates and the 0.2/0.3 m link-length ratio follow the plant model.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
BLUE = '#0072B2'
INK = '#222222'
GREY = '#777777'


def angle_arrow(ax, center, radius, start, stop, color=INK, width=.8):
    angles = np.deg2rad(np.linspace(start, stop, 100))
    xy = np.asarray(center) + radius*np.c_[np.cos(angles), np.sin(angles)]
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=width, zorder=5)
    ax.add_patch(FancyArrowPatch(xy[-5], xy[-1], arrowstyle='-|>',
                                mutation_scale=7, color=color, lw=width,
                                zorder=6))


def joint(ax, point, active):
    if active:
        ax.add_patch(Circle(point, .017, facecolor='white', edgecolor=BLUE,
                            lw=1.6, zorder=8))
        ax.add_patch(Circle(point, .008, facecolor=BLUE, edgecolor=BLUE,
                            lw=.7, zorder=9))
    else:
        ax.add_patch(Circle(point, .012, facecolor='white', edgecolor=INK,
                            lw=1.2, zorder=8))


def panel(ax, title, elbow_active):
    ax.set_aspect('equal')
    ax.set_xlim(-.25, .48)
    ax.set_ylim(-.285, .59)
    ax.axis('off')
    ax.text(.115, .565, title, ha='center', va='center', fontsize=10,
            fontweight='bold')

    l1, l2 = .2, .3
    q1, q2 = np.deg2rad([55., 95.])
    origin = np.array([0., 0.])
    elbow = l1*np.array([np.sin(q1), -np.cos(q1)])
    tip = elbow + l2*np.array([np.sin(q1+q2), -np.cos(q1+q2)])

    # Dashed links retain the same lengths at the upright target.
    ax.plot([0, 0, 0], [0, l1, l1+l2], color=GREY, lw=1.3,
            ls=(0, (4, 3)), zorder=1)
    for yy in (l1, l1+l2):
        ax.add_patch(Circle((0, yy), .009, facecolor='white',
                            edgecolor=GREY, lw=.9, zorder=2))
    ax.text(-.025, .495, 'Upright target', ha='right', va='center', fontsize=8.8)
    ax.text(-.025, .455, r'$q_\star=(\pi,0)$', ha='right', va='center', fontsize=9)

    # Fixed shoulder bearing and the actual two-link mechanism.
    ax.add_patch(Rectangle((-.07, -.034), .025, .068, facecolor='.88',
                           edgecolor=INK, lw=.8, zorder=3))
    ax.plot([-.045, 0], [0, 0], color=INK, lw=1.1, zorder=3)
    ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], color=INK,
            lw=3.1, solid_capstyle='round', zorder=4)
    ax.add_patch(Circle(tip, .008, facecolor=INK, edgecolor=INK, zorder=5))
    joint(ax, origin, not elbow_active)
    joint(ax, elbow, elbow_active)

    # q1 is measured from the downward vertical; q2 is relative to link 1.
    ax.plot([0, 0], [-.025, -.18], color=GREY, lw=.7, ls=(0, (3, 3)))
    angle_arrow(ax, origin, .068, -90, -35)
    ax.text(.031, -.103, r'$q_1$', fontsize=9.5, ha='center')
    extension = elbow + .11*np.array([np.sin(q1), -np.cos(q1)])
    ax.plot([elbow[0], extension[0]], [elbow[1], extension[1]],
            color=GREY, lw=.7, ls=(0, (3, 3)))
    angle_arrow(ax, elbow, .064, -35, 60)
    ax.text(elbow[0]+.092, elbow[1]+.005, r'$q_2$', fontsize=9.5,
            ha='center', va='center')
    ax.text(.084, -.017, r'$l_1$', fontsize=9.5)
    ax.text(.195, .035, r'$l_2$', fontsize=9.5)

    ax.annotate('', xy=(-.175, .18), xytext=(-.175, .32),
                arrowprops=dict(arrowstyle='-|>', lw=.9, color=INK))
    ax.text(-.205, .25, r'$g_0$', ha='right', va='center', fontsize=9.5)

    recovery = FancyArrowPatch(tip+np.array([.026, .016]), (.025, .515),
                              connectionstyle='arc3,rad=.32', arrowstyle='-|>',
                              color=GREY, lw=.9, mutation_scale=8)
    ax.add_patch(recovery)
    ax.text(.165, .30, 'Recovery', ha='center', fontsize=8.8)

    # Torque acts only at the selected joint; ring and text encode actuation.
    active = elbow if elbow_active else origin
    angle_arrow(ax, active, .034, 110, 355, color=BLUE, width=1.05)
    if elbow_active:
        ax.annotate('Passive shoulder', xy=(-.012, .006), xytext=(-.225, .08),
                    fontsize=8.5, ha='left', va='center',
                    arrowprops=dict(arrowstyle='-', color=INK, lw=.7,
                                    connectionstyle='angle,angleA=0,angleB=100'))
        ax.annotate('Active elbow', xy=elbow+np.array([.015, -.021]),
                    xytext=(.29, -.212), fontsize=8.8, color=BLUE,
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color=BLUE, lw=.85))
        ax.text(.29, -.252, r'$\tau_2=\gamma u$', fontsize=9.5,
                color=BLUE, ha='center')
    else:
        ax.annotate('Active shoulder', xy=(-.025, .017), xytext=(-.225, .08),
                    fontsize=8.5, ha='left', va='center', color=BLUE,
                    arrowprops=dict(arrowstyle='-', color=BLUE, lw=.85,
                                    connectionstyle='angle,angleA=0,angleB=100'))
        ax.text(-.155, -.065, r'$\tau_1=\gamma u$', color=BLUE, fontsize=9.5,
                ha='center')
        ax.annotate('Passive elbow', xy=elbow+np.array([.009, -.012]),
                    xytext=(.29, -.212), fontsize=8.8, ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color=INK, lw=.7))
    b = '(0,1)' if elbow_active else '(1,0)'
    ax.text(-.225, -.245, rf'$b={b}^\top$', fontsize=9.5, va='center')


def main():
    mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.05))
    panel(axes[0], '(a) Acrobot', True)
    panel(axes[1], '(b) Pendubot', False)
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.12, wspace=.08)
    fig.text(.5, .055, r'EGAR commands the active joint:  $|u|\leq0.5\,\mathrm{N\,m}$;'
             r'  $\tau=\gamma b u$', ha='center', va='center', fontsize=9,
             color=BLUE)
    out = HERE.parent.parent/'figures'/'simulation_systems.pdf'
    fig.savefig(out, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
