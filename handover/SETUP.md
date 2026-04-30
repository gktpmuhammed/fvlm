# Handover Setup

This repository is prepared for clean migration to a new userspace.

## Scope
- Code-only handover for training/evaluation stacks.
- No checkpoints or heavy experiment outputs required.
- Dataset files are linked with symlinks, not copied.

## Quick Start
1. Install Miniconda in the new userspace.
2. Create conda envs using scripts in `scripts/setup/`.
3. Rebuild symlinks and metadata links (see `handover/DATA_LINKS.md`).
4. Run smoke tests (see `handover/SMOKE_TESTS.md`).

## Recommended Order
1. `scripts/setup/create_env_fvlm_training_clean.sh`
2. `scripts/setup/create_env_radevalmetrics.sh`
3. `scripts/setup/create_env_ct_rate.sh`
4. Data link setup
5. Smoke tests
