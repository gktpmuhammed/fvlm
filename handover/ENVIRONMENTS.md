# Environment Rebuild

This repo expects three logical environments:
- `fvlm_training_clean`: model training/evaluation
- `radevalmetrics`: report metric evaluation
- `decompose`: report decomposition with vLLM

## Lock Files
Place generated lock files under `envs/`:
- `fvlm_training_clean.explicit.txt`
- `fvlm_training_clean.pip.txt`
- `radevalmetrics.explicit.txt`
- `radevalmetrics.pip.txt`
- `decompose.requirements.txt`

The `decompose` environment is intentionally lightweight and is built from a
small requirements file instead of the old full `ct-rate` lock.

## Freeze on source machine
Run:
```bash
bash scripts/setup/freeze_env_locks.sh
```

## Recreate on target machine
Run:
```bash
bash scripts/setup/create_env_fvlm_training_clean.sh
bash scripts/setup/create_env_radevalmetrics.sh
bash scripts/setup/create_env_decompose.sh
```

## Notes
- Do not copy old `miniconda3` directly.
- Recreate from lock files in fresh conda installation.
