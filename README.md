# dreamdock

Learning to rendezvous with tumbling space debris from pixels.

Dead satellites don't cooperate: no markers, no communication, unknown mass properties, and a tumble that can change when something hits them. dreamdock trains a stochastic world model that watches a tumbling satellite through a chaser's camera, learns how it looks and moves, and dreams many possible futures instead of one. A planner then searches those dreamed futures for an approach that is fuel-efficient and stays safe across all of them.

## How it works

1. **Simulate.** Chaser relative motion (Clohessy-Wiltshire) and target attitude (Euler's equations), with randomized hidden properties and random disturbances, rendered from the chaser's camera.
2. **Learn.** A recurrent state-space model (RSSM, as in DreamerV3) learns from frames and thrust commands only. It is never told the target's pose, spin, or inertia.
3. **Dream.** Roll the model forward to sample a distribution of futures, including the branching of unstable spins.
4. **Plan.** A cross-entropy-method planner picks thrust sequences that reach a docking point on the tumbling body across the dreamed branches.

Results are compared against true physics and a classical model-predictive controller.

## Setup

```
pip install -r requirements.txt
pytest
```
