"""
nanograd/nn.py
==============
Neural network building blocks on top of the autograd engine.

This module mirrors the structure of PyTorch's `torch.nn` but built entirely
from scratch using only our `Value` class.  It shows how higher-level
abstractions (neurons → layers → networks) emerge naturally from the basic
building block of scalar automatic differentiation.

Hierarchy
---------
Module          — abstract base: tracks parameters, enables zero-grad
  └── Neuron   — a single artificial neuron: dot product + activation
  └── Layer    — a list of Neurons sharing the same inputs
  └── MLP      — a stack of Layers (a Multi-Layer Perceptron)
"""

import random
from nanograd.engine import Value


class Module:
    """Abstract base class for all neural network modules.

    All learnable components inherit from this so they share a common
    interface:
      - `parameters()` — returns all leaf `Value` nodes that gradient descent
        should update.
      - `zero_grad()` — resets every parameter's gradient to 0 before the
        next backward pass (prevents gradient accumulation across steps).
    """

    def zero_grad(self):
        """Set the gradient of every parameter to zero.

        In a training loop, call this at the *start* of each iteration
        (before the forward pass), or just before calling `loss.backward()`.
        Failing to zero the gradients means they will accumulate across
        iterations, which is almost never desired.
        """
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """Return all learnable parameters of this module (to be overridden)."""
        return []


class Neuron(Module):
    """A single artificial neuron.

    Computes: output = activation(w · x + b)

    where `·` is the dot product between the weight vector `w` and the
    input vector `x`, and `b` is a scalar bias term.

    Parameters
    ----------
    nin : int
        Number of input features (dimensionality of `x`).
    nonlin : bool, optional
        If True (default), apply a tanh non-linearity to the weighted sum.
        Set to False for a linear neuron (e.g. in the output layer of a
        regression network).
    """

    def __init__(self, nin, nonlin=True):
        # Weights initialised uniformly in [-1, 1] — a simple heuristic that
        # works for small networks.  For deeper networks, Xavier / He init
        # would be more appropriate.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)          # bias starts at zero
        self.nonlin = nonlin

    def __call__(self, x):
        """Forward pass: compute the neuron's output for input `x`.

        Parameters
        ----------
        x : list[float | Value]
            Input vector of length `nin`.

        Returns
        -------
        Value
            The scalar output of this neuron.
        """
        # Weighted sum: Σ (wᵢ * xᵢ) + b
        # `sum(..., start)` with a Value start gives us a proper Value result.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        """Return all learnable parameters: weights + bias."""
        return self.w + [self.b]

    def __repr__(self):
        kind = "Tanh" if self.nonlin else "Linear"
        return f"{kind}Neuron({len(self.w)})"


class Layer(Module):
    """A fully connected layer: a list of Neurons, all receiving the same input.

    Parameters
    ----------
    nin : int
        Number of input features.
    nout : int
        Number of neurons in this layer (= number of output features).
    **kwargs
        Forwarded to each `Neuron` (e.g. `nonlin=False` for a linear layer).
    """

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """Forward pass through the layer.

        Parameters
        ----------
        x : list[float | Value]
            Input vector of length `nin`.

        Returns
        -------
        Value | list[Value]
            If the layer has a single neuron, returns a scalar `Value`.
            Otherwise returns a list of `Value` objects.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """Return all parameters from every neuron in the layer."""
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"


class MLP(Module):
    """Multi-Layer Perceptron — a stack of fully connected layers.

    Data flows from the first layer to the last; each layer's output is
    the next layer's input.

    Parameters
    ----------
    nin : int
        Number of input features.
    nouts : list[int]
        Sizes of each successive layer.  The last element is the number of
        output neurons.

    Example
    -------
    >>> model = MLP(3, [4, 4, 1])   # 3 inputs, two hidden layers of 4, 1 output
    >>> x = [2.0, 3.0, -1.0]
    >>> y_pred = model(x)
    """

    def __init__(self, nin, nouts):
        # Build layer sizes: [nin, nouts[0], nouts[1], ..., nouts[-1]]
        sz = [nin] + nouts
        # The last layer is linear (no activation) — common for regression /
        # when the loss function handles the final non-linearity.
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """Forward pass through every layer in sequence.

        Parameters
        ----------
        x : list[float | Value]
            Input vector of length `nin`.

        Returns
        -------
        Value | list[Value]
            Final network output.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Return all learnable parameters across every layer."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP([{', '.join(str(layer) for layer in self.layers)}])"
