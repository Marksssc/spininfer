import numpy as np

from objectives.gradient import Gradient
from objectives.regularized import RegularizedObjective

class MockArrayBackend:
    xp = np

class MockModel:
    array_backend = MockArrayBackend()

class MockObjective:
    def compute_gradient(self, model, h, J, dataset):
        return Gradient(grad_h=np.array([1.0, 1.0]), grad_J=np.array([[2.0, 2.0], [2.0, 2.0]]))
        
    def compute_value_and_gradient(self, model, h, J, dataset):
        grad = self.compute_gradient(model, h, J, dataset)
        return 10.0, grad

class MockRegularizer:
    def penalty(self, h, J, model):
        return 2.0
        
    def gradient(self, h, J, model):
        return np.array([0.1, 0.1]), np.array([[0.2, 0.2], [0.2, 0.2]])

def test_regularized_objective_gradient():
    model, dataset = MockModel(), None
    h, J = np.zeros(2), np.zeros((2, 2))
    
    obj = RegularizedObjective(objective=MockObjective(), regularizer=MockRegularizer())
    grad = obj.compute_gradient(model, h, J, dataset)
    
    assert np.allclose(grad.grad_h, np.array([0.9, 0.9]))
    assert np.allclose(grad.grad_J, np.array([[1.8, 1.8], [1.8, 1.8]]))

def test_regularized_objective_value_and_gradient():
    model, dataset = MockModel(), None
    h, J = np.zeros(2), np.zeros((2, 2))
    
    obj = RegularizedObjective(objective=MockObjective(), regularizer=MockRegularizer())
    value, grad = obj.compute_value_and_gradient(model, h, J, dataset)
    
    assert np.isclose(value, 8.0)
    assert np.allclose(grad.grad_h, np.array([0.9, 0.9]))
    assert np.allclose(grad.grad_J, np.array([[1.8, 1.8], [1.8, 1.8]]))