## Methodology

<img src="./static/images/method.png" alt="RAC Method Overview">

**Fig. 2:** Method Overview. (i) Training. To prevent latent space collapse, we freeze the VAE encoder and train only the RAC decoder; reverse-time inference then serves as encoding. (ii) State Construction. Extra channels beyond RGB are padded with 0.5, keeping the velocity field shape constant and ensuring bidirectional consistency. (iii) RAC Input. RAC takes time t and the current state as input, driving the transition from latent initialization to the target image.

<img src="./static/images/fig3.png" alt="Reconstruction is Condition Generation" style="width: 40%; height: auto;">

**Fig. 3:** Reconstruction is Condition Generation: The previous reconstructions were more accurate because they could relatively approach the manifold. The previous generations predicted variables that were often some distance away from the manifold. This is part of the reason for the past differences in the performance of generation and reconstruction. However, our method theoretically aims for perfect reconstruction, and the multi-step decoding can correct the potential variables provided by Unet or DiT. Therefore, both reconstruction and generation are significantly superior to traditional VAEs.

<img src="./static/images/fig4.png" alt="Conceptual and empirical views of RAC generation trajectories">

**Fig. 4:** Conceptual and empirical views of RAC generation trajectories. Left: a conceptual illustration of trajectory-based generation from latent space to image space. Right: sampled state trajectories projected into a 2D PCA space.
