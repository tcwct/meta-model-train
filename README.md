# Meta Model Train

This directory is the cleaned, standalone project for the `meta model` experiments.
It is designed to be the future working area for:

- local development
- GitHub synchronization
- server-side training on the `mpp-meta` conda environment

## Project Layout

```text
meta-model-train/
  environment.yml
  pyproject.toml
  README.md
  .gitignore
  docs/
  scripts/
  src/
```

## What Is Included

- the 2D diffusion data generator used by the toy training task
- the 2.0 minimal architecture family model
- smoke training and family training entry scripts
- meta-dataset aggregation and plotting utilities
- a server sync note that matches the SSH-over-443 workflow already validated

## Current 2.0 Setting

- architecture tokens are restricted to `M` and `A`
- `M` is a residual 2-layer MLP with zero-initialized second linear layer
- `A` is a residual axial attention block with zero-initialized output projections
- the default depth is `8`
- the default embedding dimension is `8`
- architecture syntax constraints from the earlier `L/A/R` version are removed

## What Is Not Included

- old checkpoints
- large output folders
- historical local scratch files

## Recommended Workflow

1. Work inside this directory.
2. Optionally install the local package in editable mode:

```bash
pip install -e .
```

3. Push this directory to a dedicated GitHub repository.
4. On the server, clone with `git@github.com:<user>/<repo>.git`.
5. Activate the environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mpp-meta
```

6. Run the first smoke training:

```bash
python scripts/train_minimal_arch_smoke.py --device cuda --max_steps 50
```

## First Commands

Run the smallest end-to-end self-check:

```bash
python scripts/run_smoke_pipeline.py --device cpu
```

Or use the unified project CLI:

```bash
python scripts/meta_model_cli.py smoke-pipeline --device cpu
```

On the server, the fastest first-run path is:

```bash
bash scripts/server_quickstart.sh
```

Check whether the current machine can really run the project environment:

```bash
python scripts/check_environment.py
```

Or:

```bash
python scripts/meta_model_cli.py check-env
```

Enumerate a small architecture subset:

```bash
python scripts/enumerate_architectures.py --sample_size 8
```

Train a small family:

```bash
python scripts/train_minimal_arch_family.py \
  --architecture_csv artifacts/sampled_architectures_L8_n8_seed25.csv \
  --device cuda \
  --max_steps 100
```

Build the meta dataset:

```bash
python scripts/build_meta_dataset.py \
  --family_dir outputs/toy_diffusion/meta_model_family/<family_name>
```

## Notes

- Python 3.11 is required.
- The validated server environment uses PyTorch 2.5 with CUDA 12.1.
- A minimal package definition is included in [pyproject.toml](C:/Users/11599/Desktop/本科生科研/毕业论文/multiple_physics_pretraining-main/meta-model-train/pyproject.toml) so this folder can behave like an independent repo.
- `outputs/` and generated artifacts are intentionally ignored by git.
- The active training setting is documented in [docs/2.0_setting说明.md](C:/Users/11599/Desktop/本科生科研/毕业论文/multiple_physics_pretraining-main/meta-model-train/docs/2.0_setting说明.md).
- A server-first walkthrough is in [docs/首跑说明.md](C:/Users/11599/Desktop/本科生科研/毕业论文/multiple_physics_pretraining-main/meta-model-train/docs/首跑说明.md).
- The 2.0 full-coverage 2000-step server workflow is documented in [docs/服务器2.0全量2000步训练说明.md](C:/Users/11599/Desktop/本科生科研/毕业论文/multiple_physics_pretraining-main/meta-model-train/docs/服务器2.0全量2000步训练说明.md).
- The 2.1 full-coverage 16000-step server workflow is documented in [docs/服务器2.1全量16000步训练说明.md](C:/Users/11599/Desktop/本科生科研/毕业论文/multiple_physics_pretraining-main/meta-model-train/docs/服务器2.1全量16000步训练说明.md).
- A one-command local or server self-check is available in [scripts/run_smoke_pipeline.py](C:/Users/11599/Desktop/本科生科研/毕业论文/multiple_physics_pretraining-main/meta-model-train/scripts/run_smoke_pipeline.py).
- A unified command entrypoint is available in [scripts/meta_model_cli.py](C:/Users/11599/Desktop/本科生科研/毕业论文/multiple_physics_pretraining-main/meta-model-train/scripts/meta_model_cli.py).
- A concise repository bootstrap checklist is in [docs/建仓与首训清单.md](C:/Users/11599/Desktop/本科生科研/毕业论文/multiple_physics_pretraining-main/meta-model-train/docs/建仓与首训清单.md).
