# EGAR reproduction code

Acrobot and Pendubot experiments for *Energy-Guided Adaptive Rollout for Control Recovery under Actuation Loss*. Pretrained policy weights and the initial states used in the paper are included.

## Installation

Use Python 3.11 on Linux x86_64. The experiments run on the CPU and do not require a display. Run the commands below from this directory.

```bash
conda create -n egar python=3.11 pip -y
conda activate egar
python -m pip install -r requirements-baselines.txt
```

`requirements-baselines.txt` includes `requirements.txt` and the additional Python dependencies for the public NMPC and iLQR controllers. To run EGAR without these native baselines, install `requirements.txt` alone.

For the full comparison, install the build tools and native libraries. On Ubuntu/Debian:

```bash
sudo apt-get install build-essential cmake git libeigen3-dev libyaml-cpp-dev
```

Alternatively, install them in the conda environment:

```bash
conda install -c conda-forge c-compiler cxx-compiler cmake make eigen yaml-cpp -y
```

Then build the public baselines:

```bash
python install_baselines.py
```

This command downloads the sources to `external/` and builds the iLQR extension, acados libraries, and template renderer. Installation requires internet access. The experiments run locally once installation is complete.

## Running the experiments

Check policy loading and short simulations of both systems:

```bash
python reproduce.py smoke
```

Run the complete experiment set, followed by statistical analysis and figure generation:

```bash
python reproduce.py all
```

The full run contains thousands of 20 s simulated episodes, including repeated NMPC solves. Wall-clock time depends on the machine and solver configuration.

Rerunning the command resumes unfinished work. Completed tasks and saved NMPC episodes are reused. For a fresh run, use a new output directory:

```bash
python reproduce.py all --output results_fresh
```

Results are saved under the selected output directory, which defaults to `results/`:

| Directory | Contents |
|---|---|
| `figures/` | PDF figures |
| `tables/` | Tables 3–9 as CSV and detailed statistics as JSON |
| `work/` | Simulation trajectories and intermediate calculations |
| `logs/` | Execution logs |
| `completed/` | Task completion markers for resuming a run |
| `build/` | Generated solver code |

The figure scripts produce the system schematic and experimental results (Figures 2–16). Figure 1 is the conceptual controller diagram. Tables 1–2 describe the algorithm and simulation settings, which are specified in the source and model-parameter file.

### Individual experiment groups

Use `python reproduce.py <group>` to run one group. The order below is also used by `all`.

| Group | Experiments | Prerequisites |
|---|---|---|
| `nominal` | Both systems: EGAR, terminal-cost ablations, clipped policy, Xin-Kaneda and PFL | Core Python dependencies |
| `gain` | Gain loss with noisy feedback, gain sweep, candidate costs and rankings | Core Python dependencies |
| `continuation` | Policy, zero-input and held-command continuations | `nominal` for paired analysis |
| `multiseed` | Five noise realizations for Acrobot static/gain-step tests and Pendubot gain-step tests | `gain` for the seed-51 comparison |
| `public` | Public acados NMPC and iLQR-MPC; external controllers under gain loss and noise | Native baselines |
| `shared` | RTI and energy NMPC with fixed or online-calibrated models; solver development trajectories | Native baselines |
| `solver` | SQP accuracy comparisons and tighter-tolerance closed-loop runs | `shared` |
| `certificate` | Local LQR design, interval certificate and local responses | Core Python dependencies |
| `supplementary` | Computation time, MuJoCo checks and dissipative operating limit | Native baselines and MuJoCo |
| `figures` | Statistical summaries, CSV tables and PDF figures | All experiment groups except `supplementary` |

List the individual tasks or select one by name:

```bash
python reproduce.py all --list
python reproduce.py nominal --only authority_calibration_aligned_core_confirmation
python reproduce.py multiseed --only pendubot_step_51_dynamics
```

Dependent groups must use the same output directory. Run each task in only one process at a time to avoid overwriting its files.

## Simulation settings and reference results

The state is `[q1, q2, dq1, dq2]`. The shoulder angle `q1` is measured from the downward vertical, and `q2` is the elbow angle relative to the first link. Acrobot actuates the elbow; Pendubot actuates the shoulder.

The command limit is 0.5 N m. Plant integration and local feedback use a 0.002 s step, and EGAR replans every 0.05 s. Recovery requires angular errors below 0.15 rad and speeds below 0.6 rad/s for 0.5 s. Unrecovered episodes are assigned 20 s in the capped mean, or 19 s for recovery after the gain change at 1 s.

The nominal experiments should reproduce the following results:

| System | Variant | Recovery | Mean capped time (s) |
|---|---|---:|---:|
| Acrobot | EGAR | 64/64 | 3.709 |
| Acrobot | Critic terminal | 64/64 | 5.105 |
| Acrobot | Energy term removed | 54/64 | 9.240 |
| Pendubot | EGAR | 64/64 | 5.688 |
| Pendubot | Tuned critic terminal | 40/64 | 10.922 |
| Pendubot | Energy term removed | 22/64 | 14.924 |

With noisy feedback and reduced Acrobot gain, calibrated EGAR recovers 64/64 states with a mean capped time of 4.968 s; the fixed predictor recovers 30/64. The local certificate uses gain interval `[0.8, 1.25]`, sampling period 0.002 s, and P-norm radius 0.0016.

Experiments use the supplied paired initial states, public AR-EAPO weights and observation normalization, and the baseline settings selected during development. Policy training and parameter searches are not required. The main tables and figures use the final event-aligned controller. Supplementary MuJoCo and dissipative-limit checks use the development-controller configurations reported for those checks.

Floating-point libraries and asynchronous solver scheduling can affect trajectories and NMPC results. Computation times depend on hardware and system load. Recovery times measure simulated plant time; computation delays are not added to the plant simulation.

## Source files and external dependencies

`reproduce.py` runs the experiments; `install_baselines.py` installs the native baselines. Controller, simulation, certificate and analysis code is in `src/`. Policy weights, initial states, mechanical parameters and the iLQR initialization trajectory are in `assets/`.

Public source repositories:

- [DFKI double_pendulum](https://github.com/dfki-ric-underactuated-lab/double_pendulum).
- [acados](https://github.com/acados/acados).

To use existing installations, set `DOUBLE_PENDULUM_DIR` and `ACADOS_SOURCE_DIR` to their source directories. The runner sets the Python and shared-library paths. The reported experiments used CasADi 3.8.2, which triggers a compatibility warning in this acados release.

The DFKI policy weights, model parameters and initial trajectory are distributed under the BSD 3-Clause license in `assets/LICENSE-double-pendulum.txt`. Policy weights are stored as inference-only arrays with unchanged numerical values. Downloaded upstream sources retain their license files.
