# Sharp Thresholds in Reward Hacking

A research program treating reward hacking, monitor evasion, and collective intent
drift as **five faces of one object**: the detection edge of a partial-channel
spiked-covariance monitoring problem. The faces differ only in which property of
that edge one reads — its position, its steepness, or the curvature of the approach —
and whether one reads it statically, in the training dynamics, or across a population.

A single message runs through all five: **the detector sharpness that makes a
non-adaptive monitor powerful is exactly what makes the resulting failure abrupt,
irreversible, and hard to forewarn.** And one mathematical object — the **cusp**
catastrophe — organizes both ends of the scale: bistability is *born* at a cusp as a
single agent's monitor sharpens, and *annihilated* at a cusp as a population's
vulnerability diversifies.

All results are proved and verified in simulation on a deliberately simple surrogate;
each paper states its scope, evidence status, and the conditions that would falsify
the mechanism. The mathematical machinery is classical (random-matrix edge theory,
catastrophe theory, early-warning signals, threshold cascades) and imported with
credit; the contributions are the mapping to reward hacking, a closed-form threshold,
a random-matrix microfoundation, and a set of actionable laws.

Author: **Guangyu Wang** (billgywang@gmail.com)

## Papers

| Folder | Paper | Role |
|---|---|---|
| `overview/`   | *One Threshold, Five Faces* (compact overview) | the program in 5 pages |
| `synthesis/`  | *One Threshold, Five Faces: From Detection Cliffs to Controlled Reward-Hacking Dynamics* | self-contained synthesis; core theorem with proof, modules, evidence table |
| `catastrophe/`| *The Cliff Becomes a Catastrophe: Saddle-Node Reward Hacking under Sharp Monitoring* | **diagnosis** — the single-agent fold, closed-form threshold, BBP microfoundation |
| `controller/` | *A Closed-Loop Controller for Reward Hacking: Critical Slowing Down and Nonsmooth Damping* | **control** — CSD sensor, nonsmooth actuator, switched-system stability, rate law |
| `collective/` | *Collective Intent Drift: Weakest-Link Cascades and a Disorder-Induced Critical Endpoint* | **population** — mean-field avalanche, weakest-link law, cusp endpoint |

Each paper folder is self-contained: the `.tex`, its figures, and the compiled `.pdf`
are co-located, so `pdflatex <paper>.tex` (run 2–3×) compiles in place.

## The five faces

| Face | Reads | Lens | Result |
|---|---|---|---|
| 1 wall        | position ρ\* | static          | recovery cliff / covering slope |
| 2 catastrophe | steepness σ = Θ(d^{1/3}) | dynamic, 1 agent | fold at σ > σ_c; sharper is worse |
| 3 sensor      | curvature κ → 0 | dynamic, online | critical slowing down warns where detection erodes |
| 4 actuator    | moves σ_c | dynamic, control | nonsmooth damping, no detection-power cost |
| 5 population  | σ_c distributed | dynamic, many | weakest-link κ\* ∼ −ln f; cusp endpoint at CV_c |

Faces 2 and 5 share **one cusp**: bistability is born (single agent, σ↑σ_c) and
annihilated (population, CV↑CV_c) through the same codimension-two singularity.

## Code

Reference implementations (CPU-only, `numpy`/`scipy`) reproducing the figures and
reported numbers. Organized by paper under `code/`:

- `code/catastrophe/` — `intent_drift.py`, `duality_dynamics.py`, `learned_detector_toy.py`
- `code/controller/`  — `csd_probe.py`, `controller_loop.py`, `actuator_probe.py`, `ratelaw_probe.py`
- `code/collective/`  — `collective_drift.py`, `collective_deep.py`, `collective_endpoint.py`

Each script prints the quantities it reproduces. Example:

```bash
python code/controller/controller_loop.py     # open vs closed loop past the fold
python code/collective/collective_endpoint.py # disorder-induced critical endpoint
python code/catastrophe/learned_detector_toy.py  # hysteresis with a learned detector
```

## Scope

All numerical validation is on the surrogate, not on a trained agent. The papers
state exactly what is proved, what is a companion result, what is simulated, and what
is heuristic (see the evidence-status tables), and describe the controlled
reinforcement-learning study with a learned detector needed to test the mechanism
beyond the surrogate.

## License

Released under CC BY 4.0 (see `LICENSE`). If you use this work, please cite the
relevant paper and attribute Guangyu Wang.
