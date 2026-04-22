# Local RL Trading Agent — Plan

A structured, milestone-driven plan you can hand to Claude Code to
build a local, paper-trading-capable RL trading system for Indian
equities. One end-to-end path first, then specialize.

## How to use this

1. Read `00_overview.md` first. Especially the **Honest caveats**
   section — the plan pushes back on a few parts of the original spec
   and explains why.
2. Review `08_open_decisions.md` and override any defaults you
   disagree with before starting. (Shorting? Data frequency? Universe
   size? All listed there.)
3. Open Claude Code in an empty repo, and paste the **Prompt seed**
   from `07_roadmap.md` Milestone 0 to start.
4. Proceed milestone by milestone. Each milestone has explicit
   acceptance criteria that must pass before the next one starts.

## Files in this plan

| File | Purpose |
| --- | --- |
| `00_overview.md` | Goal, architecture, design principles, caveats, assumptions |
| `01_setup_and_infrastructure.md` | Repo layout, deps, Docker, Postgres, MLflow, tooling |
| `02_data_pipeline.md` | Sources, universe, alignment with masks, features |
| `03_environment.md` | `PanelTradingEnv` spec: obs, action, reward, costs |
| `04_models.md` | TCN encoder + Hetero GAT + actor-critic heads |
| `05_training.md` | PPO, walk-forward eval, metrics, overfitting guards |
| `06_paper_broker.md` | DB schema, `Broker` ABC, paper simulator, live stub |
| `07_roadmap.md` | 11 milestones (M0–M10) with acceptance criteria and prompt seeds |
| `08_open_decisions.md` | Defaults chosen; override before starting |

## Key design departures from the original brief

Summarized so you know what to argue with:

1. **Non-existent stocks are masked, not zero-filled.** Zero-fill would
   leak a fake regime-shift signal. The mask is part of the observation
   and the action space.
2. **Graph edges are learned by backprop through a GAT, not set by an
   RL action.** Same interpretability outcome ("learned sector
   relations"), vastly better credit assignment.
3. **One hierarchical actor-critic in v1, not two agents.** A second
   agent is an open door in v2, not a starting point.
4. **LLM/sentiment is offline-only.** Static weekly embeddings if at
   all. No per-step LLM calls.
5. **Baselines are first-class citizens.** The RL agent has to beat
   equal-weight and momentum after real Indian transaction costs
   before any result is reported.
6. **"Real-time adaptation" is walk-forward retraining on a schedule,
   not online gradient updates during live trading.**

## Hardware assumptions

- NVIDIA RTX 5090 (Blackwell), needs CUDA 12.8+ PyTorch build.
- Intel Core Ultra 9 285K (Arrow Lake).
- Plenty of RAM and fast local SSD.

The plan is not compute-bound; model sizes are modest on purpose so
the first result ships quickly.

## If something fails an acceptance criterion

- Do not skip ahead.
- Roll back the commit and diagnose.
- The invariants (especially in M3 alignment and M4 env) catch silent
  correctness bugs that you would otherwise notice only in returns —
  at which point the cost to debug is 10× higher.
