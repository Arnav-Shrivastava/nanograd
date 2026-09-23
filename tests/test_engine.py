"""
tests/test_engine.py
====================
Sanity checks for the nanograd autograd engine.

These tests verify that:
  1. Forward passes compute the correct numeric values.
  2. Backward passes compute gradients that match hand-derived analytical
     solutions (and PyTorch where applicable).
  3. Edge cases (multiple uses of the same variable, chains of operations)
     are handled correctly.

Run with:  pytest tests/test_engine.py -v
"""

import math
import pytest
from nanograd.engine import Value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_close(a, b, tol=1e-6, msg=""):
    """Assert that two floats are within `tol` of each other."""
    assert abs(a - b) < tol, f"{msg} | expected {b:.8f}, got {a:.8f}"


# ---------------------------------------------------------------------------
# 1. Forward-pass tests
# ---------------------------------------------------------------------------

class TestForwardPass:
    """Verify that Value arithmetic produces correct numeric outputs."""

    def test_addition(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_addition_with_scalar(self):
        a = Value(2.0)
        c = a + 3.0
        assert c.data == 5.0

    def test_radd(self):
        a = Value(2.0)
        c = 3.0 + a
        assert c.data == 5.0

    def test_multiplication(self):
        a = Value(3.0)
        b = Value(-4.0)
        c = a * b
        assert c.data == -12.0

    def test_multiplication_with_scalar(self):
        a = Value(5.0)
        c = a * 2.0
        assert c.data == 10.0

    def test_subtraction(self):
        a = Value(7.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 4.0

    def test_negation(self):
        a = Value(5.0)
        assert (-a).data == -5.0

    def test_division(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert_close(c.data, 2.0)

    def test_power(self):
        a = Value(2.0)
        c = a ** 3
        assert_close(c.data, 8.0)

    def test_tanh(self):
        a = Value(0.0)
        assert_close(a.tanh().data, 0.0)
        b = Value(1.0)
        assert_close(b.tanh().data, math.tanh(1.0))

    def test_relu(self):
        pos = Value(3.0)
        neg = Value(-3.0)
        assert pos.relu().data == 3.0
        assert neg.relu().data == 0.0

    def test_exp(self):
        a = Value(1.0)
        assert_close(a.exp().data, math.e)

    def test_chained_operations(self):
        # (2 * 3) + (4 * 5) = 6 + 20 = 26
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        d = Value(5.0)
        out = a * b + c * d
        assert_close(out.data, 26.0)


# ---------------------------------------------------------------------------
# 2. Backward-pass tests (gradient checks)
# ---------------------------------------------------------------------------

class TestBackwardPass:
    """Verify that gradients match analytically-derived values."""

    def test_addition_gradients(self):
        # out = a + b  →  da = 1, db = 1
        a = Value(2.0)
        b = Value(3.0)
        out = a + b
        out.backward()
        assert_close(a.grad, 1.0, msg="da for a+b")
        assert_close(b.grad, 1.0, msg="db for a+b")

    def test_multiplication_gradients(self):
        # out = a * b  →  da = b = -4, db = a = 3
        a = Value(3.0)
        b = Value(-4.0)
        out = a * b
        out.backward()
        assert_close(a.grad, -4.0, msg="da for a*b")
        assert_close(b.grad,  3.0, msg="db for a*b")

    def test_power_gradient(self):
        # out = x^3  →  dx = 3 * x^2 = 3 * 4 = 12 (x=2)
        x = Value(2.0)
        out = x ** 3
        out.backward()
        assert_close(x.grad, 12.0, msg="dx for x^3 at x=2")

    def test_tanh_gradient(self):
        # out = tanh(x)  →  dx = 1 - tanh(x)^2
        x = Value(0.5)
        out = x.tanh()
        out.backward()
        expected = 1 - math.tanh(0.5) ** 2
        assert_close(x.grad, expected, msg="dx for tanh at x=0.5")

    def test_relu_gradient_positive(self):
        x = Value(3.0)
        out = x.relu()
        out.backward()
        assert_close(x.grad, 1.0, msg="dx for relu at x=3 (positive)")

    def test_relu_gradient_negative(self):
        x = Value(-3.0)
        out = x.relu()
        out.backward()
        assert_close(x.grad, 0.0, msg="dx for relu at x=-3 (negative)")

    def test_chain_rule_simple(self):
        # out = (a + b) * c
        # d_out/da = c = 3, d_out/dc = a + b = 5
        a = Value(2.0)
        b = Value(3.0)
        c = Value(3.0)
        out = (a + b) * c
        out.backward()
        assert_close(a.grad, 3.0, msg="da chain rule")
        assert_close(b.grad, 3.0, msg="db chain rule")
        assert_close(c.grad, 5.0, msg="dc chain rule")

    def test_variable_used_twice(self):
        # out = a * a = a^2  →  da = 2*a = 4 (a=2)
        # With accumulation: each use of `a` in the *graph contributes
        # a gradient, so da += b.data * out.grad twice.
        a = Value(2.0)
        out = a * a
        out.backward()
        assert_close(a.grad, 4.0, msg="da for a*a at a=2")

    def test_complex_expression_matches_pytorch(self):
        """
        Replicate the key neuron example from the tutorial notebooks and
        verify our gradients match hand-calculated values.

        Expression: o = tanh(x1*w1 + x2*w2 + b)
        Inputs: x1=2, x2=0, w1=-3, w2=1, b=6.8813735870195432
        Expected output: o ≈ 0.7071 (≈ tanh(0.8814))
        """
        x1 = Value(2.0)
        x2 = Value(0.0)
        w1 = Value(-3.0)
        w2 = Value(1.0)
        b  = Value(6.8813735870195432)

        n = x1 * w1 + x2 * w2 + b
        o = n.tanh()
        o.backward()

        # Analytically: do/dn = 1 - tanh(n)^2 ≈ 0.5
        # do/dw1 = do/dn * x1 = 0.5 * 2 = 1.0  → BUT note sign convention:
        # In the canonical example from the tutorial, w1.grad = 1.0
        # because w1 = -3 and x1 = 2, but ∂o/∂w1 = x1 * (1 - o^2) = 2 * 0.5

        t = o.data  # tanh output
        local_grad = 1 - t ** 2  # gradient of tanh at n

        assert_close(o.data, math.tanh(0.8813735870195432), tol=1e-5,
                     msg="forward pass output")
        assert_close(w1.grad, x1.data * local_grad, tol=1e-5,
                     msg="w1 gradient")
        assert_close(x1.grad, w1.data * local_grad, tol=1e-5,
                     msg="x1 gradient")
        assert_close(b.grad, local_grad, tol=1e-5,
                     msg="bias gradient")


# ---------------------------------------------------------------------------
# 3. Gradient accumulation (multiple paths through the graph)
# ---------------------------------------------------------------------------

class TestGradientAccumulation:
    """When a Value node appears multiple times in a graph, its gradient
    should accumulate contributions from every path."""

    def test_multipath_gradient(self):
        # out = a * b + a * c  →  da = b + c = 7
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        out = a * b + a * c
        out.backward()
        assert_close(a.grad, b.data + c.data, msg="da multipath")

    def test_zero_grad_resets(self):
        """Simulates a second training step — grads must be zeroed first."""
        from nanograd.nn import Neuron
        neuron = Neuron(2)
        x = [Value(1.0), Value(2.0)]

        # First forward+backward
        out = neuron(x)
        out.backward()
        grads_first = [p.grad for p in neuron.parameters()]

        # Zero grads then repeat
        neuron.zero_grad()
        out2 = neuron(x)
        out2.backward()
        grads_second = [p.grad for p in neuron.parameters()]

        # Because input and weights are the same, grads should be identical
        for g1, g2 in zip(grads_first, grads_second):
            assert_close(g1, g2, msg="grads should be identical after zero_grad")


# ---------------------------------------------------------------------------
# 4. Neural-network smoke test
# ---------------------------------------------------------------------------

class TestMLP:
    """End-to-end test: train a tiny MLP for a few steps and verify loss decreases."""

    def test_loss_decreases(self):
        from nanograd.nn import MLP

        random_seed = 42
        import random
        random.seed(random_seed)

        model = MLP(3, [4, 4, 1])

        xs = [
            [2.0,  3.0, -1.0],
            [3.0, -1.0,  0.5],
            [0.5,  1.0,  1.0],
            [1.0,  1.0, -1.0],
        ]
        ys = [1.0, -1.0, -1.0, 1.0]

        # Initial loss
        ypred = [model(x) for x in xs]
        loss0 = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # Training loop — 20 steps
        for _ in range(20):
            ypred = [model(x) for x in xs]
            loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= 0.05 * p.grad

        ypred_final = [model(x) for x in xs]
        loss_final = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred_final))

        assert loss_final.data < loss0.data, \
            f"Loss should decrease: {loss0.data:.4f} → {loss_final.data:.4f}"
