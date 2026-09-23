"""
nanograd/engine.py
==================
The autograd engine — the heart of nanograd.

This module defines the `Value` class, a scalar wrapper that records every
arithmetic operation it participates in and uses that record (a dynamic
computational graph) to compute exact gradients via reverse-mode automatic
differentiation (backpropagation).

Key concepts implemented here
------------------------------
* **Forward pass** — ordinary Python arithmetic, but each operation stores
  its inputs so we can retrace the computation later.
* **Backward pass** — starting from the output node (loss), gradients are
  propagated backwards through every node in topological order using the
  chain rule.
* **Topological sort** — ensures every node's gradient has been fully
  accumulated before its contribution is sent further back.
* **Operator overloading** — Python dunder methods (`__add__`, `__mul__`, …)
  let `Value` objects behave exactly like plain numbers in expressions.
"""

import math


class Value:
    """A scalar value in the computational graph.

    Wraps a Python float and tracks:
      - `data`   : the scalar value itself.
      - `grad`   : the gradient of the final output w.r.t. this value
                   (populated after calling `.backward()`).
      - `_prev`  : the set of `Value` nodes that were used to produce this one.
      - `_op`    : a string label for the operation that created this node
                   (e.g. '+', '*', 'tanh') — useful for visualisation.
      - `_backward` : a closure that, when called, distributes `self.grad`
                      backwards to `self._prev` according to the chain rule.

    Parameters
    ----------
    data : float | int
        The numeric value to wrap.
    _children : tuple[Value, ...], optional
        The input `Value` nodes that produced this node.
    _op : str, optional
        Human-readable label for the creating operation.
    label : str, optional
        An optional name for this node (handy for graph visualisation).
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # The backward function — populated by each operation's factory method.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # ------------------------------------------------------------------
    # Core arithmetic operations
    # Each method:
    #   1. Computes the forward value.
    #   2. Defines a `_backward` closure that applies the chain rule for
    #      this specific operation.
    #   3. Returns a new `Value` that links back to its inputs.
    # ------------------------------------------------------------------

    def __add__(self, other):
        """self + other

        Local gradient:  d(out)/d(self) = 1,  d(out)/d(other) = 1
        Chain rule:      self.grad  += 1.0 * out.grad
                         other.grad += 1.0 * out.grad
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """self * other

        Local gradient:  d(out)/d(self) = other.data
                         d(out)/d(other) = self.data
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """self ** other  (other must be int or float, not a Value)

        Local gradient:  d(out)/d(self) = other * self^(other-1)
        """
        assert isinstance(other, (int, float)), \
            "Only int/float powers are supported; 'other' cannot be a Value."
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    def tanh(self):
        """Hyperbolic tangent activation.

        Forward:   t = tanh(x) = (e^2x - 1) / (e^2x + 1)
        Backward:  d(tanh)/dx = 1 - tanh(x)^2  = 1 - t^2
        """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """Rectified Linear Unit activation.

        Forward:   out = max(0, x)
        Backward:  d(relu)/dx = 1 if x > 0 else 0
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        """Exponential function  e^x.

        Forward:   out = e^x
        Backward:  d(e^x)/dx = e^x = out.data
        """
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # d/dx(e^x) = e^x, and we already stored that as out.data
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        """Natural logarithm  ln(x).

        Forward:   out = ln(x)
        Backward:  d(ln x)/dx = 1/x
        """
        assert self.data > 0, "log is only defined for positive values."
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out

    # ------------------------------------------------------------------
    # Reverse operations (handle `scalar op Value` expressions)
    # ------------------------------------------------------------------

    def __radd__(self, other):   # other + self
        return self + other

    def __rmul__(self, other):   # other * self
        return self * other

    def __rsub__(self, other):   # other - self
        return other + (-self)

    def __rtruediv__(self, other):   # other / self
        return other * self ** -1

    # ------------------------------------------------------------------
    # Derived operations (implemented in terms of the core ops above)
    # ------------------------------------------------------------------

    def __neg__(self):           # -self
        return self * -1

    def __sub__(self, other):    # self - other
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backward(self):
        """Compute gradients for all `Value` nodes in the computational graph.

        Algorithm
        ---------
        1. Build a topological ordering of all ancestor nodes (DFS post-order).
        2. Set this node's gradient to 1.0 (dL/dL = 1).
        3. Walk the topological order in reverse, calling each node's
           `_backward` closure to propagate gradients one step at a time.

        Why topological order?
        ~~~~~~~~~~~~~~~~~~~~~~
        A node may be consumed by multiple downstream nodes (e.g. a weight
        used in many neurons).  We must be sure that *all* gradient
        contributions from downstream nodes have been accumulated into a
        node's `.grad` before we call its own `_backward`; topological order
        guarantees exactly that.
        """
        # Step 1 — topological sort via depth-first search
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)   # append *after* processing children → post-order

        build_topo(self)

        # Step 2 — seed gradient at the root
        self.grad = 1.0

        # Step 3 — propagate gradients in reverse topological order
        for node in reversed(topo):
            node._backward()
