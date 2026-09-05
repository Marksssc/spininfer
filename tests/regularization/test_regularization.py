import numpy as np
from objectives.regularization import L1Regularizer, L2Regularizer, CompositeRegularizer, default_regularization_strength


class MockArrayBackend:
    xp = np

class MockModel:
    array_backend = MockArrayBackend()

def test_default_regularization_strength():
    assert np.isclose(default_regularization_strength(100), 0.01)

def test_l2_regularizer_penalty_and_gradient():
    model = MockModel()
    h = np.array([1.0, 2.0])
    J = np.array([[1.0, -1.0], [-1.0, 1.0]])
    
    reg = L2Regularizer(strength=0.1, apply_to_h=True, apply_to_J=True)

    # Penalty: 0.5 * 0.1 * sum(h^2) + 0.5 * 0.1 * sum(J^2)
    # h^2 sum = 5.0, J^2 sum = 4.0 -> (0.25 + 0.20) = 0.45
    assert np.isclose(reg.penalty(h, J, model), 0.45)
    
    grad_h, grad_J = reg.gradient(h, J, model)
    assert np.allclose(grad_h, np.array([0.1, 0.2]))
    assert np.allclose(grad_J, np.array([[0.1, -0.1], [-0.1, 0.1]]))

def test_l1_regularizer_penalty_and_gradient():
    model = MockModel()
    h = np.array([1.0, -2.0])
    J = np.array([[0.5, -0.5], [-0.5, 0.5]])
    
    reg = L1Regularizer(strength=0.2, apply_to_h=True, apply_to_J=True)

    # Penalty: 0.2 * sum(abs(h)) + 0.2 * sum(abs(J))
    # abs(h) sum = 3.0, abs(J) sum = 2.0 -> (0.6 + 0.4) = 1.0
    assert np.isclose(reg.penalty(h, J, model), 1.0)
    
    grad_h, grad_J = reg.gradient(h, J, model)
    assert np.allclose(grad_h, np.array([0.2, -0.2]))
    assert np.allclose(grad_J, np.array([[0.2, -0.2], [-0.2, 0.2]]))

def test_regularizer_apply_flags():
    model = MockModel()
    h = np.array([1.0, 2.0])
    J = np.array([[1.0, -1.0], [-1.0, 1.0]])
    
    reg_J_only = L2Regularizer(strength=0.1, apply_to_h=False, apply_to_J=True)
    assert np.isclose(reg_J_only.penalty(h, J, model), 0.20)
    
    grad_h, grad_J = reg_J_only.gradient(h, J, model)
    assert np.allclose(grad_h, np.zeros_like(h)) 
    assert np.allclose(grad_J, 0.1 * J)
    
    reg_h_only = L1Regularizer(strength=0.1, apply_to_h=True, apply_to_J=False)
    assert np.isclose(reg_h_only.penalty(h, J, model), 0.30)
    
    grad_h, grad_J = reg_h_only.gradient(h, J, model)
    assert np.allclose(grad_h, 0.1 * np.sign(h))
    assert np.allclose(grad_J, np.zeros_like(J))

def test_composite_regularizer():
    model = MockModel()
    h = np.array([1.0, -2.0])
    J = np.array([[1.0, -1.0], [-1.0, 1.0]])
    
    reg1 = L2Regularizer(strength=0.1, apply_to_h=True, apply_to_J=False)
    reg2 = L1Regularizer(strength=0.2, apply_to_h=False, apply_to_J=True)
    comp_reg = CompositeRegularizer(regularizers=[reg1, reg2])
    
    expected_penalty = reg1.penalty(h, J, model) + reg2.penalty(h, J, model)
    assert np.isclose(comp_reg.penalty(h, J, model), expected_penalty)
    
    grad_h, grad_J = comp_reg.gradient(h, J, model)
    expected_grad_h = reg1.gradient(h, J, model)[0] + reg2.gradient(h, J, model)[0]
    expected_grad_J = reg1.gradient(h, J, model)[1] + reg2.gradient(h, J, model)[1]
    
    assert np.allclose(grad_h, expected_grad_h)
    assert np.allclose(grad_J, expected_grad_J)