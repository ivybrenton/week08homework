from scipy.integrate import quad
from scipy.special import comb

class BayesFactor:
    def __init__(self, n, k):
        """
        Initialize the BayesFactor object.
        
        :param n: Total number of trials
        :param k: Number of successes
        """
        if not isinstance(n, int) or not isinstance(k, int):
            raise TypeError("n and k must be integers!")
        if n < 0 or k < 0:
            raise ValueError("n and k must be non-negative!")
        if k > n:
            raise ValueError("k cannot be greater than n!")
        
        self.n = n
        self.k = k

    def likelihood(self, theta):
        """
        Compute the binomial likelihood of observing k successes in n trials.
        
        :param theta: Probability of success
        :return: Binomial likelihood
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("theta must be numeric!")
        if not (0 <= theta <= 1):
            raise ValueError("theta must be between 0 and 1!")
        
        return comb(self.n, self.k) * (theta**self.k) * ((1 - theta)**(self.n - self.k))

    def evidence_slab(self):
        """
        Compute the marginal likelihood under the slab prior (Uniform(0, 1)).
        
        :return: Marginal likelihood
        """
        # Prior is Uniform(0, 1), so p(theta | Slab) = 1 for theta in [0, 1]
        # Marginal likelihood = integral of (likelihood * prior) d_theta from 0 to 1
        
        func = lambda theta: self.likelihood(theta)
        result, _ = quad(func, 0, 1)
        return result

    def evidence_spike(self):
        """
        Compute the marginal likelihood under the spike prior (Uniform(0.47, 0.53)).
        
        :return: Marginal likelihood
        """
        # Prior is Uniform(0.47, 0.53), so p(theta | Spike) = 1 / (0.53 - 0.47) = 1 / 0.06
        # Marginal likelihood = integral of (likelihood * prior) d_theta from 0.47 to 0.53
        
        prior_density = 1 / 0.06
        func = lambda theta: self.likelihood(theta) * prior_density
        result, _ = quad(func, 0.47, 0.53)
        return result

    def bayes_factor(self):
        """
        Compute the Bayes Factor (BF = evidence_spike / evidence_slab).
        
        :return: Bayes Factor
        """
        return self.evidence_spike() / self.evidence_slab()