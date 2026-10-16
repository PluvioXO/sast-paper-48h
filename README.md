# SAST: 48-hour, six-GPU experiment runner

Run from the project root. The default protocol targets a Linux machine with six RTX 2080 GPUs and stops 48 hours after the first `run` invocation. It uses one independent worker per GPU; SLI and pooled GPU memory are unnecessary. The code runs in FP32 with small microbatches and records the effective batch size separately.

## Clone

```bash
git clone https://github.com/PluvioXO/sast-paper-48h.git
cd sast-paper-48h
```

## Install and launch

Use Python 3.11 or 3.12 on a recent Linux distribution, a compatible NVIDIA driver, an SSD with about 100 GB available, and preferably at least 32 GB RAM. Install PyTorch separately so pip selects its CUDA build. The pinned CUDA 12.6 wheel is listed in the [official PyTorch index](https://download.pytorch.org/whl/cu126/torch/); see also [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/requirements-paper-linux.txt

python run_paper_experiments.py doctor --gpus 0,1,2,3,4,5 --data /mnt/ssd/sast_data
python run_paper_experiments.py plan
python run_paper_experiments.py run \
  --gpus 0,1,2,3,4,5 --microbatch 4 --threads 2 \
  --data /mnt/ssd/sast_data --output /mnt/ssd/sast_48h --budget-hours 48
```

GPU indices are logical indices after any `CUDA_VISIBLE_DEVICES` mask. `doctor` executes a CUDA convolution and backward pass on every selected GPU, catching incompatible wheels/drivers before downloading data. It also reports free GPU memory and disk space. No CUDA toolkit compilation, NCCL setup, or distributed training launch is required.

The launch command automatically downloads data, verifies the adapters' expected counts, builds frame caches, benchmarks representative training and inference, and freezes the final plan. Downloading/preprocessing counts against the 48-hour deadline. Existing caches are reused after checksum verification.

## What fits into the two-day protocol

The requested maximum matrix is **84 tuning fits, 140 confirmation fits, five causal experiment jobs, and 28 diagnostic jobs**, plus a cheap analytic toy sweep. Each causal job contains one common pretraining fit and 12 continuation arms. The first seed also includes three short adaptation fits. Before research training, measured throughput chooses 3–5 confirmation seeds and a common fraction of the requested epoch budgets. It retains longer training before adding seeds, applies a 1.5× timing safety factor, and reserves 20% of the remaining time. Every method in a case receives the same resulting budget rule.

| Experiment group | Default work |
| --- | --- |
| Confirmation and generality | DVS Gesture on two- and four-layer CNNs; CIFAR10-DVS on a two-layer CNN; SHD on a two-layer spiking MLP. Full configured training partitions and full test sets. |
| Fair competing methods | Two ordinary updates, SAST, same-batch SAM, standard same-batch ASAM, direct hard-forward surrogate-gradient training, state-margin regularization, and SAST with state-margin regularization. Three balanced candidate configurations and one pilot seed per method/case; fresh confirmation seeds. |
| Causal continuation | Identical DVS Gesture CNN checkpoints across the 2×2 SAST × margin design and controls: ordinary training, two updates, SAM, both ASAM batch variants, two averaged batches, random norm-matched perturbations, rho zero, and hard-forward fine-tuning. Optimizer state is reset identically for continuation arms. |
| Controlled mathematics | Explicit tiny recurrent counterexamples over surrogate slope, leak, time horizon, and perturbation radius; smooth loss, gradient, Hessian, hard accuracy, and numerical interval bounds. |
| Transfer and representations | Componentwise signed-reset-Jacobian and class-specific-readout bounds versus a loose scalar bound; raw and calibrated conversion damage/rescue; per-layer CKA; feature collapse; stable ridge heads fitted on smooth/raw/calibrated features and cross-evaluated. |
| Noise and geometry | Fixed-checkpoint gradient noise versus batch size/rho; saved normalized directions; small measured loss/accuracy planes at each method's own checkpoint; short multistart neighborhood searches; four-step Hessian estimates. |
| Cheap deployment diagnostics | Output/reset hardening homotopy; fixed/global/per-layer calibration comparisons; frame corruption tests; short continuous-input PGD attacks; temperature calibration, ECE/Brier scores; dense PyTorch inference timing and memory. |

Large training ablation grids, input-gradient double-backprop training, SGD retraining grids, dense landscape grids, long adversarial searches, and new full N-MNIST training are **excluded from the default launch**. These choices keep the two-day plan focused. Advanced method/configuration support is exercised by `smoke`, but is not scheduled by the default research preset.

The timing estimate cannot guarantee that every job finishes: downloads, CPU contention, GPU clocks, and thermal throttling vary. The runner rejects a measured plan that cannot fit even at its minimum settings. A hard deadline stops outstanding workers and saves a partial report; it never fabricates missing results or silently treats partial work as complete. The clock persists across interruptions and resumes. An expired run can still be inspected with `status` and `report`; it does not receive another 48 hours automatically.

## Data and scientific protocol

- DVS Gesture uses the official train/test partitions and a stratified validation partition inside training. CIFAR10-DVS uses a declared, deterministic stratified 80/10/10 split because it has no official train/test split. SHD uses the official test set and a speaker-disjoint validation partition inside training. SHD's official test set is not described as entirely unseen-speaker data: both model-unseen and officially unseen speaker subsets are reported separately. The [SHD dataset description](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) gives the source protocol.
- Vision preprocessing reuses the repository's existing verified adapters: 16 equal-duration bins and 64×64 spatial resolution for DVS Gesture/CIFAR10-DVS. SHD uses 64 equal-duration bins over 700 channels. Frame normalization is per recording; temporal duration information is therefore normalized, and the cache metadata records this choice.
- The default neurons retain the current paper's leak 0.5, threshold 1, and arctan slope 1. The compact search covers learning rates 0.0003/0.001/0.003 and SAST/SAM/ASAM radii 0.05/0.2/0.4, with balanced joint candidates rather than an exhaustive Cartesian grid.
- Pilot and confirmation seeds are disjoint. Pilot training maps only train/validation caches. Checkpoints minimize validation CE (hard-forward CE for the direct-hard baseline). Method selection uses calibrated validation accuracy, then CE, then candidate index. Test metrics never select settings/checkpoints. Data preparation itself constructs test caches, but pilot evaluation cannot access them through the loader.
- Threshold calibration uses a fixed balanced validation subset of up to 256 recordings, then reports full validation accuracy. Global search is followed by one coordinate pass. Boundary winners trigger expanded search, bounded by positive scales 0.015625–8; remaining final boundary hits are flagged. Test thresholds are frozen. Confidence temperatures and ridge strengths are also selected on validation data only.
- All causal arms share one saved pretraining endpoint per seed, with identical continuation learning rate/weight decay and matched forward/backward example budgets for the two-pass/two-update comparisons. Margin beta and SAST rho come from their designated validation pilots. The unadapted common model is evaluated too. The short adaptation comparison uses 10% of the base epoch budget on the first seed.
- Effective batches are sampled with replacement. Independent ascent/descent streams may overlap by chance. Gradients accumulate weighted microbatch means, including a shorter last microbatch. SAST stops the perturbation gradient, restores the original weights, then applies AdamW using the second gradient. ASAM uses `abs(weight)+0.01` for weight tensors and unit bias scale. Standard ASAM reuses the batch; its independent-batch variant is a separate causal control.
- Primary accuracy uses the full test set. Expensive diagnostics use fixed balanced subsets: up to 128 test recordings, 512 probe-training recordings, 256 probe-validation recordings, 32 transfer-bound recordings, and 32 geometry-validation recordings. Diagnostics use the first confirmation seed. These are descriptive samples, not additional independent training replications.
- Corruptions operate on cached normalized frames: cell/bin masking, injected background occupancy, temporal permutation/reversal/shift, and small spatial shifts. They are not raw-event deletion experiments. PGD perturbs continuous frames in `[0,1]`; hard-forward attacks use the declared arctan surrogate gradient. Neither experiment establishes certified event robustness.
- Transfer bounds are a posteriori real-arithmetic error bounds evaluated numerically in float64 using hard trajectories. The report records violations and prediction-agreement coverage; zero observed violations is not formal floating-point certification or proof of a new theorem. Numerical Hessian/neighborhood estimates and 2D slices likewise do not certify global sharpness.

## Resume, inspect, and customize

```bash
# Rerun the identical launch command before the original deadline to resume.
python run_paper_experiments.py status --output /mnt/ssd/sast_48h
python run_paper_experiments.py report --output /mnt/ssd/sast_48h

# Write and inspect a configuration before starting a NEW run directory.
python run_paper_experiments.py init paper_48h.json
python run_paper_experiments.py plan --config paper_48h.json

# Run selected stages; subsequent commands must use the same configuration/data/output.
python run_paper_experiments.py run --stages toy,tune \
  --data /mnt/ssd/sast_data --output /mnt/ssd/sast_48h
python run_paper_experiments.py run --stages confirm,causal,analysis,report \
  --data /mnt/ssd/sast_data --output /mnt/ssd/sast_48h
```

Research runs create `requested_protocol.json`, hardware execution records, actual benchmarks, `budget_plan.json`, the effective `protocol.json`, and a frozen `selection.json`. Configurations and source hashes are immutable within a run directory. Completed artifacts are checksum-verified. A scheduler lock prevents two launches from owning the same output; dataset locks protect shared caches. Each job has a log and failure record.

Epoch checkpoints include the model, optimizer, best model, both sampling streams, perturbation RNG, and PyTorch RNG. An interrupted epoch repeats from its preceding checkpoint. An interrupted causal job retains completed arms. Files are saved atomically where they establish progress; an incomplete analysis artifact never counts as a completed job.

If a worker runs out of GPU memory, stop/resume with `--microbatch 2` or `1`. This keeps the effective batch size; execution records note the change. Floating-point summation can change slightly across microbatch sizes, GPU/software versions, or hardware. Use the same microbatch/software throughout when exact repeatability matters. Use a new output directory for scientific configuration/source changes.

## Outputs and local verification

`reports/primary.csv`, `paired_comparisons.csv`, `factorial.csv`, `causal.csv`, and `compute_audit.csv` summarize completed work. `summary.json` retains detailed diagnostics and a completeness flag. `accuracy.pdf/png` and per-case landscape comparison PDFs/PNGs are standalone figures; every method is plotted around its own saved checkpoint. Raw logits, sample IDs, ridge heads, directions, bounds, checkpoints, calibration histories, and logs remain under `jobs/`. The existing manuscript is not modified automatically.

Confidence intervals use paired training seeds, not individual test examples. Holm correction covers the primary SAST-versus-two-updates family across cases. Other contrasts and power calculations are exploratory. Three to five seeds may leave wide intervals; more experiments cannot guarantee acceptance, and the 48-hour limit constrains statistical power.

```bash
# No downloads/CUDA required; outputs are prominently marked synthetic.
python run_paper_experiments.py smoke --output /tmp/sast_smoke --microbatch 2

# Optional developer checks.
python -m pip install pytest
python -m pytest -q tests/test_paper_suite.py
```

The tests check the existing CNN recurrence, independent explicit two-pass update references, finite-difference recurrent gradients, uneven microbatch accumulation, restoration after interrupted perturbations, exact CPU epoch resume, speaker splitting, numerical transfer-bound coverage, collapse-safe ridge/CKA behavior, immutable results, timing-only budget selection, and partial reports. CPU smoke covers every supported method and all execution stages. Six-GPU Linux execution and the actual 48-hour runtime require verification on the target machine.
