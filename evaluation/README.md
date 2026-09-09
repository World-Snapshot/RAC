# RAC controlled evaluation

This directory implements three complementary comparisons requested after review. Generated data,
weights, third-party repositories, and results are ignored by Git; the protocol and adapters are
versioned.

## 1. Official-checkpoint reconstruction benchmark

Evaluate all available ImageNet-256 checkpoints on exactly the same prepared images and metric
implementation. Quantitative rows currently supported are RAC/TAESD, SD-VAE, SSDD, FlexTok, and
SEMANTICIST. DiTo publishes code but no official checkpoint; Epsilon-VAE has no public official
implementation/checkpoint; Diffusion Autoencoders publishes checkpoints for other datasets, not
ImageNet. Those three remain in the mechanism table but must not receive fabricated or OOD numbers.

Official defaults are retained for architecture-specific inference:

- RAC: 4 forward/reverse Euler steps.
- SSDD-S-F8C4: official F8C4 checkpoint, 8 decoder steps.
- FlexTok d12-d12 ImageNet: 256 tokens, 20 steps, CFG 7.5, norm guidance.
- SEMANTICIST-L: 256 slots, 250 steps, CFG 3.0.

Prepare a smoke set (replace the source with the mounted validation directory):

```bash
python prepare_inputs.py --source /path/to/imagenet/val --output data/smoke4 --count 4
```

Export reconstructions (run from this directory):

```bash
python run_rac.py --inputs data/smoke4 --output results/rac/recon --device cuda:0
python run_ssdd.py --inputs data/smoke4 --output results/ssdd/recon --device cuda:0
python run_sdvae.py --inputs data/smoke4 --output results/sdvae/recon --device cuda:0
/research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python run_flextok.py \
  --inputs data/smoke4 --output results/flextok/recon --device cuda:0
/research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python run_semanticist.py \
  --inputs data/smoke4 --output results/semanticist/recon --device cuda:0
```

Evaluate each output with `evaluate_reconstructions.py`, then combine tables with
`aggregate_results.py`. The existing local 1K subset may be used only as a pilot. The paper table
requires ImageNet validation 50K.

## 2. Controlled from-scratch comparison

See `training/README.md`. This is RAC-1x versus SSDD-S-F8C4 and DiTo-B-F8C4 under matched data and
budget, with both fixed-budget and converged results. Native training losses are not ranked across
methods.

## 3. Latent-shift robustness

Inject `z_sigma = z + sigma * std(z) * epsilon`, with paired seeds and
`sigma in {0, .1, .2, .3, .5}`. This comparison is restricted to continuous F8C4 methods (RAC,
SD-VAE, SSDD, and a trained DiTo-F8C4). FlexTok's discrete FSQ sequences and SEMANTICIST's 1D slots
do not share this latent geometry and are therefore excluded rather than perturbed artificially.

## Interpretation guardrail

The mature local RAC checkpoint is Tiny-RAC (about 1.46M parameters), whereas SSDD-S has 48.22M
parameters. Tiny-RAC validates the pipeline and offers a parameter-efficiency point; it is not the
primary capacity-matched row. The formal comparison needs a newly trained, converged RAC-1x model.
