# Comprehensive Course on Variational Autoencoders (VAEs)

I struggled for a long time to properly understand Variational Autoencoders. Most resources assumed prior knowledge of mathematical concepts, and when I tried to learn them separately, they were often explained in contexts unrelated to VAEs, such as GANs or other deep learning models. I created this course to give anyone with similar goals a clear and foundational path to understanding VAEs from the ground up.

**Author:** Adam Mazouar
**Date:** December 2, 2025  
**Version:** 1.0

## Course Structure

This course is organized into multiple files for better navigation:

1. **[Part 1: Foundations & Theory](Part1_Foundations.md)** - Mathematical foundations, probability review, generative models
2. **[Part 2: The VAE Framework](Part2_VAE_Framework.md)** - ELBO derivation, reparameterization, KL divergence
3. **[Part 3: Implementation](Part3_Implementation.md)** - PyTorch and TensorFlow code, training loops
4. **[Part 4: Advanced Topics](Part4_Advanced.md)** - β-VAE, VQ-VAE, hierarchical VAEs, extensions
5. **[Part 5: Practical Guide](Part5_Practical.md)** - Training tips, debugging, failure modes
6. **[Appendix A: Derivations](Appendix_A_Derivations.md)** - Complete mathematical derivations
7. **[Appendix B: Code Repository](code/)** - All runnable code examples
8. **[Appendix C: Exercise Solutions](Appendix_C_Solutions.md)** - Detailed solutions
9. **[Appendix D: LaTeX Source](Appendix_D_LaTeX.tex)** - Typeset equations
11. **[Cheat Sheet](CheatSheet.md)** - One-page summary
12. **[Glossary](Glossary.md)** - All terms and symbols
13. **[FAQ](FAQ.md)** - Frequently asked questions
14. **[Quiz](Quiz.md)** - Self-assessment with answers

## Prerequisites

- Linear algebra (matrices, eigenvalues, vector calculus)
- Probability theory (distributions, expectations, Bayes' theorem)
- Deep learning basics (neural networks, backpropagation, PyTorch or TensorFlow)
- Python programming

## Quick Start

```bash
# Install dependencies
pip install torch torchvision tensorflow numpy matplotlib scikit-learn

# Run minimal VAE example
cd code
python minimal_vae_pytorch.py
```

## Learning Path

See the detailed 12-week study schedule in Part 5.

---

**Citation:** If you use this course material, please cite:
```
@misc{vae_course_2025,
  title={Course on Variational Autoencoders},
  author={Adam Mazouar},
  date={december 2025}
}
```