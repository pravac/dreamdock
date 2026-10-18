# dreamdock

A stochastic world model that learns how a tumbling dead satellite looks and moves from camera frames, dreams a distribution over its possible futures, and plans a safe, fuel-efficient approach that holds up across those futures.

## Why learning

With known state and known physics the future is deterministic, and a network is just a slow integrator. Learning matters only under uncertainty, so the simulator hides it:

1. **Hidden properties**: inertia tensor (including intermediate-axis cases), energy dissipation (fuel slosh)
2. **Hidden state**: only pixels are observed; pose and spin are never given
3. **Random disturbances**: impacts arriving at random times (Poisson), solar radiation pressure torque

Intermediate-axis spin (the Dzhanibekov effect) is unstable: tiny perturbations change when the body flips. A good model has to branch.

## Data

Everything is simulated: CW relative motion plus Euler attitude dynamics, rendered from the chaser's camera. The model trains on frames and actions only; true states are logged for scoring. Real data is used for validation only: SPEED+ (sim-to-real check on the encoder), ADRAS-J imagery (qualitative), MMT-9 light curves (spin-rate priors).

## Plan

### Ramp week (Oct 19 to 25)
1. Mon. Clohessy-Wiltshire relative motion sim (`dreamdock/cw.py`)
2. Tue. Torque-free rigid body attitude (Euler's equations, quaternions), then dissipation and impacts
3. Wed. VAE on rendered frames, derive and implement the ELBO
4. Thu. Ha and Schmidhuber style next-latent prediction with an MDN-RNN
5. Fri. DreamerV3 (torch) on a toy env, study the RSSM
6. Sat. CEM planner in latent space (PlaNet-style MPC) on the toy env
7. Sun. Frame rendering pipeline, frozen spec and metrics

### Build (from Oct 26)
1. Dataset: frames, chaser actions, true states, with randomized hidden parameters and disturbances
2. Train a stochastic RSSM world model on frames plus actions
3. Risk-aware latent CEM planner: reach a docking point on the tumbling body, safe across dreamed branches
4. Classical baseline: CW + Euler MPC with nominal parameters (known state, unknown hidden properties)
5. Evaluation and plots

## Metrics
1. Prediction error vs rollout horizon, learned model vs true physics
2. Calibration of dreamed futures: coverage and CRPS against the true future distribution
3. Surprise response: belief update speed after an impact
4. Approach success rate, collision rate, and delta-v: learned planner vs baseline

## Run

```
pip install -r requirements.txt
pytest
```
