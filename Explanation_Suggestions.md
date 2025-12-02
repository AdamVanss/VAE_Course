If you are a reader please add suggestions for what needs to be explained better.

In part1:

in 1.1 What is a Generative Model?
    you should tell about some intuition on what a lower bound of Pθ(x) means

in 1.2 The Data Distribution vs Model Distribution
    in The Empirical Distribution phat_data(x)
        you should explain more dirac delta function, how can it "place all probability mass exactly on our observed data points." and why is it good for us to do it.

in 1.4 Probability Distributions Review
    For a multivariate Gaussian with diagonal covariance (independent dimensions):
        you should explain what is a multivariable gaussian just enough intuition to keep up with the rest
    For KL Divergence
        you should explain why is it an expectation, general intuition on why it looks that way

in 1.5 Maximum Likelihood Estimation
    you should explain with a simple mathematical exemple what is argmin / argmax of a single variable function, and the variable under the max or min
    further explain Interpretation: Negative log-likelihood is related to bits needed to encode data

1.6 The Intractability Problem
    you should give a definition of Intractability

in part2 VAE Framework
Jensen inequality not explained before-hand