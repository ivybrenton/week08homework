from scipy.integrate import quad
from scipy.special import comb

class BayesFactor:
    def __init__(self, n, k):
        if not isinstance(n, int) or not isinstance(k, int):
            raise TypeError("n and k must be integers!")
        if n < 0 or k < 0:
            raise ValueError("n and k must be non-negative!")
        if k > n:
            raise ValueError("k cannot be greater than n!")
        
        self.n = n
        self.k = k

    def likelihood(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("theta must be numeric!")
        if not (0 <= theta <= 1):
            raise ValueError("theta must be between 0 and 1!")
        
        return comb(self.n, self.k) * (theta**self.k) * ((1 - theta)**(self.n - self.k))

    def evidence_slab(self):
        # Slab prior: Uniform(0, 1). Prior PDF is 1 for 0 <= theta <= 1.
        # Marginal likelihood = integral from 0 to 1 of (likelihood(theta) * 1) d_theta
        result, _ = quad(self.likelihood, 0, 1)
        return result

    def evidence_spike(self):
        # Spike prior: Uniform(0.47, 0.53). 
        # Prior PDF is 1 / (0.53 - 0.47) = 1 / 0.06
        # Marginal likelihood = integral from 0.47 to 0.53 of (likelihood(theta) * (1 / 0.06)) d_theta
        prior_pdf = 1.0 / 0.06
        integrand = lambda theta: self.likelihood(theta) * prior_pdf
        result, _ = quad(integrand, 0.47, 0.53)
        return result

    def bayes_factor(self):
        slab = self.evidence_slab()
        spike = self.evidence_spike()
        return spike / slab
