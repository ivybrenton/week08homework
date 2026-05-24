import unittest

from bayes_factor import BayesFactor

class TestInitialization(unittest.TestCase):
    def test_valid_initialization(self): #Ensure object state is consistent after construction.
        bf = BayesFactor(10, 5)
        self.assertEqual(bf.n, 10)
        self.assertEqual(bf.k, 5)

    def test_invalid_type(self): #Reject invalid n and k values (type and range checks).
        with self.assertRaises(TypeError):
            BayesFactor(10.5, 5)

    def test_negative_values(self): #Reject invalid n and k values (type and range checks).
        with self.assertRaises(ValueError):
            BayesFactor(-1, 5) 
    
    def test_k_greater_than_n(self): #Reject impossible binomial states (e.g., k > n).
        with self.assertRaises(ValueError):
            BayesFactor(5, 10)

    def test_invalid_type_error_message(self): #Reject invalid n and k values (type and range checks).
        with self.assertRaisesRegex(TypeError, "n and k must be integers!"):
            BayesFactor(10.5, 5)

    def test_theta_error_message(self): #Tests verify exception messages for multiple error cases.
        bf = BayesFactor(10, 5)
        with self.assertRaisesRegex(ValueError, "theta must be between 0 and 1!"):
            bf.likelihood(1.5)


class TestLikelihood(unittest.TestCase):
    def setUp(self):  #Include at least one fixture or shared setup pattern.
        self.bf = BayesFactor(10, 5)

    def test_likelihood_known_value(self): #likelihood(theta) behaves sensibly at obvious points.
        bf = BayesFactor(10, 5)
        result = bf.likelihood(0.5)
        self.assertAlmostEqual(result, 0.24609375, places=7)

    def test_likelihood_invalid_theta_type(self): #Invalid theta values are handled predictably.
        bf = BayesFactor(10, 5)
        with self.assertRaises(TypeError):
            bf.likelihood("bad")

    def test_likelihood_invalid_theta_range(self): #Invalid theta values are handled predictably.
        bf = BayesFactor(10, 5)
        with self.assertRaises(ValueError):
            bf.likelihood(1.5)


class TestEvidence(unittest.TestCase):
    def test_evidence_slab_non_negative(self): #Tests verify consistency between outputs and your documented mathematical choices.
        bf = BayesFactor(10, 5)
        result = bf.evidence_slab()
        self.assertGreater(result, 0)

    def test_evidence_spike_non_negative(self): #Tests verify consistency between outputs and your documented mathematical choices.
        bf = BayesFactor(10, 5)
        result = bf.evidence_spike()
        self.assertGreater(result, 0)

    def test_spike_evidence_greater_than_slab_for_balanced_data(self): #Tests verify consistency between outputs and your documented mathematical choices.
        bf = BayesFactor(10, 5)
        self.assertGreater(bf.evidence_spike(), bf.evidence_slab())


class TestBayesFactor(unittest.TestCase):
    def test_bayes_factor_greater_than_one_for_balanced_data(self): #bayes_factor() handles edge cases without crashing. / Tests verify consistency between outputs and your documented mathematical choices.
        bf = BayesFactor(10, 5)
        self.assertGreater(bf.bayes_factor(), 1) 

    def test_bayes_factor_known_value(self): #bayes_factor() handles edge cases without crashing. / Tests verify consistency between outputs and your documented mathematical choices.
        bf = BayesFactor(10, 5)
        self.assertAlmostEqual(bf.bayes_factor(), 2.6908590485749495, places=6)


class TestReturnTypes(unittest.TestCase):
    def test_return_types(self): #Methods return the expected data type/shape.
        bf = BayesFactor(10, 5)
        self.assertIsInstance(bf.likelihood(0.5), float)
        self.assertIsInstance(bf.evidence_slab(), float)
        self.assertIsInstance(bf.evidence_spike(), float)
        self.assertIsInstance(bf.bayes_factor(), float)


class TestIntentionalFailure(unittest.TestCase):
    
    def test_intentionally_failing_test(self): 
        bf = BayesFactor(10, 5)
        self.assertAlmostEqual(bf.bayes_factor(), 2.6908590485749495, places=6)

