# Reading List and Resources

## Essential Papers (Must Read)

### Foundational

1. **Kingma, D.P. & Welling, M. (2013)**
   - *Auto-Encoding Variational Bayes*
   - arXiv:1312.6114
   - **THE** original VAE paper. Read this first.
   - Key contributions: ELBO, reparameterization trick, VAE formulation

2. **Rezende, D.J., Mohamed, S., & Wierstra, D. (2014)**
   - *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*
   - ICML 2014
   - Independent concurrent work on VAEs
   - Key contributions: SGVB, stochastic gradient variational Bayes

### Tutorials and Reviews

3. **Doersch, C. (2016)**
   - *Tutorial on Variational Autoencoders*
   - arXiv:1606.05908
   - Excellent pedagogical introduction
   - Good for building intuition

4. **Kingma, D.P. (2017)**
   - *Variational Inference and Deep Learning: A New Synthesis*
   - PhD Thesis, University of Amsterdam
   - Comprehensive treatment of VAEs and variational inference
   - Highly recommended for deep understanding

---

## Important Extensions

### β-VAE and Disentanglement

5. **Higgins, I., et al. (2017)**
   - *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*
   - ICLR 2017
   - Introduces β-VAE for disentanglement

6. **Burgess, C.P., et al. (2018)**
   - *Understanding disentangling in β-VAE*
   - arXiv:1804.03599
   - Analysis of β-VAE mechanisms

7. **Kim, H. & Mnih, A. (2018)**
   - *Disentangling by Factorising*
   - ICML 2018
   - FactorVAE, alternative disentanglement approach

### Improved Bounds and Inference

8. **Burda, Y., Grosse, R., & Salakhutdinov, R. (2015)**
   - *Importance Weighted Autoencoders*
   - ICLR 2016
   - IWAE: tighter bounds using importance sampling

9. **Rezende, D.J. & Mohamed, S. (2015)**
   - *Variational Inference with Normalizing Flows*
   - ICML 2015
   - Flexible posteriors using normalizing flows

10. **Kingma, D.P., et al. (2016)**
    - *Improving Variational Inference with Inverse Autoregressive Flow*
    - NeurIPS 2016
    - IAF for efficient, flexible posteriors

### Discrete and Vector Quantized

11. **van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017)**
    - *Neural Discrete Representation Learning*
    - NeurIPS 2017
    - VQ-VAE: discrete latents, avoids posterior collapse

12. **Razavi, A., van den Oord, A., & Vinyals, O. (2019)**
    - *Generating Diverse High-Fidelity Images with VQ-VAE-2*
    - NeurIPS 2019
    - Hierarchical VQ-VAE for high-quality images

### Hierarchical VAEs

13. **Sønderby, C.K., et al. (2016)**
    - *Ladder Variational Autoencoders*
    - NeurIPS 2016
    - Hierarchical VAE with ladder connections

14. **Vahdat, A. & Kautz, J. (2020)**
    - *NVAE: A Deep Hierarchical Variational Autoencoder*
    - NeurIPS 2020
    - State-of-the-art hierarchical VAE
    - *Verify current literature for more recent work*

### Prior Learning

15. **Tomczak, J.M. & Welling, M. (2018)**
    - *VAE with a VampPrior*
    - AISTATS 2018
    - Learnable mixture prior

### Addressing Posterior Collapse

16. **Bowman, S.R., et al. (2016)**
    - *Generating Sentences from a Continuous Space*
    - CoNLL 2016
    - KL annealing for text VAEs

17. **He, J., et al. (2019)**
    - *Lagging Inference Networks and Posterior Collapse in Variational Autoencoders*
    - ICLR 2019
    - Analysis and solutions for collapse

---

## Related Areas

### Normalizing Flows

18. **Papamakarios, G., et al. (2021)**
    - *Normalizing Flows for Probabilistic Modeling and Inference*
    - JMLR
    - Comprehensive survey of normalizing flows

### Diffusion Models

19. **Ho, J., Jain, A., & Abbeel, P. (2020)**
    - *Denoising Diffusion Probabilistic Models*
    - NeurIPS 2020
    - Foundation of modern diffusion models

20. **Kingma, D.P., et al. (2021)**
    - *Variational Diffusion Models*
    - NeurIPS 2021
    - Connection between VAEs and diffusion

### GANs (for comparison)

21. **Goodfellow, I., et al. (2014)**
    - *Generative Adversarial Nets*
    - NeurIPS 2014
    - Original GAN paper

---

## Textbooks

22. **Bishop, C.M. (2006)**
    - *Pattern Recognition and Machine Learning*
    - Springer
    - Chapter 10: Approximate Inference (EM, variational inference)
    - Chapter 12: Continuous Latent Variables (PCA, factor analysis)

23. **Murphy, K.P. (2022)**
    - *Probabilistic Machine Learning: Advanced Topics*
    - MIT Press
    - Chapters on deep generative models
    - Free online: probml.github.io

24. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**
    - *Deep Learning*
    - MIT Press
    - Chapter 20: Deep Generative Models
    - Free online: deeplearningbook.org

---

## Online Resources

### Tutorials and Blog Posts

- **Lilian Weng's Blog:** "From Autoencoder to Beta-VAE"
  - lilianweng.github.io/posts/2018-08-12-vae/
  - Excellent visual explanations

- **Jeremy Jordan:** "Variational Autoencoders"
  - jeremyjordan.me/variational-autoencoders/
  - Good practical introduction

- **Jaan Altosaar:** "Tutorial - What is a variational autoencoder?"
  - jaan.io/what-is-variational-autoencoder-vae-tutorial/
  - Interactive visualizations

### Code Repositories

- **PyTorch Examples:**
  - github.com/pytorch/examples/tree/main/vae
  - Official PyTorch VAE example

- **Keras Examples:**
  - keras.io/examples/generative/vae/
  - TensorFlow/Keras implementation

- **Pyro (Probabilistic Programming):**
  - pyro.ai/examples/vae.html
  - VAE using Pyro probabilistic programming

- **PyTorch-VAE Collection:**
  - github.com/AntixK/PyTorch-VAE
  - Collection of many VAE variants
  - *Check for updates and forks*

### Courses

- **CS 228: Probabilistic Graphical Models (Stanford)**
  - Contains variational inference material
  
- **CS 236: Deep Generative Models (Stanford)**
  - Comprehensive coverage of VAEs and related models
  - Notes available online

---

## Suggested Reading Order

### Week 1-2: Foundations
1. Kingma & Welling (2013) - Original paper
2. Doersch (2016) - Tutorial
3. Bishop Ch. 10 - Background on variational inference

### Week 3-4: Understanding Deeply
4. Kingma PhD Thesis (2017) - Comprehensive treatment
5. Burda et al. (2015) - IWAE

### Week 5-6: Extensions
6. Higgins et al. (2017) - β-VAE
7. van den Oord et al. (2017) - VQ-VAE
8. Rezende & Mohamed (2015) - Flows

### Week 7-8: Advanced
9. Vahdat & Kautz (2020) - Hierarchical VAEs
10. Ho et al. (2020) - Diffusion (for comparison)

---

## Keeping Up to Date

**Conferences to follow:**
- NeurIPS (December)
- ICML (July)
- ICLR (May)
- AISTATS (April)

**ArXiv categories:**
- stat.ML (Machine Learning)
- cs.LG (Learning)
- cs.CV (Computer Vision)

**Search terms:**
- "variational autoencoder"
- "latent variable models"
- "deep generative models"
- "representation learning"

*Note: The field moves quickly. Verify current literature for the latest advances, especially in hierarchical VAEs and their connection to diffusion models.*


