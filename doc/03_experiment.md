## Experiments

<img src="./static/images/fig5.png" alt="The State Space of RAC is a More Organized Representation Space">

**Fig. 5:** The State Space of RAC is a More Organized Representation Space. (a) We used PCA to conduct a visual analysis of the state space that we encoded. This state space represents the state before the final output of the encoder. We can clearly observe that the training of RAC has made the original latent representation more orderly and clean. (b) Comparison of gFID-50K on ImageNet 256×256 [2] across different VAE configurations. We report results for the baseline methods (REPA [33], REPA-E [15, 22] and EQ-VAE [12]), comparing each against our approach trained under identical settings. Lower gFID-50K indicates better generation quality. RAC consistently outperforms the corresponding baselines across all VAE settings.

<img src="./static/images/fig6.png" alt="Qualitative Results on Imagenet 256 x 256 using RAC">

**Fig. 6:** Qualitative Results on Imagenet 256 × 256 using RAC. It can achieve significant results in the state reconstruction mode with just 30k steps.

<img src="./static/images/image.png" alt="Existing baseline latent manifolds and RAC latent regularity">

**Fig. 11:** The existing baseline latent manifolds show architecture-dependent anisotropy, where principal components are dominated by either stochastic high-frequency texture or coarse low-frequency blocks instead of semantic structure. RAC consistently re-allocates variance to spatially coherent components, yielding cleaner, more content-aligned latent geometry while preserving scene identity, which suggests improved cross-architecture latent regularity.
