## Abstract

In this paper, we propose a Rectified Flow Auto Coder (RAC) inspired by Rectified Flow to replace the traditional VAE:
(1) it achieves multi-step decoding by applying the decoder to flow timesteps, and its decoding path is straight and
correctable, enabling step-by-step refinement;
(2) the model inherently supports bidirectional inference, where the decoder serves as the encoder through time reversal
(hence Coder rather than encoder or decoder), reducing parameter count by nearly 41%;
(3) this generative decoding method improves generation quality since the model can correct latent variables along the path,
partially addressing the reconstruction-generation gap.
Experiments show that RAC surpasses SOTA VAEs in both reconstruction and generation with approximately 70% lower computational cost.
