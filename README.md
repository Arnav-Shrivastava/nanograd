# 🧠 nanograd

> *A tiny scalar-valued autograd engine built from scratch — understanding neural networks at the atomic level.*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](tests/)

---

## 📖 Table of Contents

1. [Project Context](#-project-context)
2. [Why I Built This](#-why-i-built-this)
3. [What I Learned](#-what-i-learned)
4. [Quickstart](#-quickstart)
5. [Theory: From Basic to Mastery](#-theory-from-basic-to-mastery)
   - [The Basics: Forward Passes & Derivatives](#the-basics-forward-passes--derivatives)
   - [Intermediate: The Chain Rule & Computational Graphs](#intermediate-the-chain-rule--computational-graphs)
   - [Mastery: Topological Sort, Neurons & Gradient Descent](#mastery-topological-sort-neurons--gradient-descent)
6. [Package Structure](#-package-structure)
7. [Running the Tests](#-running-the-tests)
8. [Notebooks](#-notebooks)
9. [Acknowledgements](#-acknowledgements)

---

## 🎯 Project Context

This is a **learning project** — built from absolute scratch, one line at a time, as I worked through Andrej Karpathy's legendary YouTube tutorial *"The spelled-out intro to neural networks and backpropagation: building micrograd"*.

The goal was simple but ambitious: **understand neural networks at the foundational level**, not just use them. Instead of `import torch` and calling `.backward()` as if it were magic, I wanted to *write* that `.backward()` myself.

`nanograd` is the result: a 200-line pure-Python library that can:
- Wrap scalar values in a graph-aware `Value` object
- Perform **automatic differentiation** (backprop) through any expression built from those values
- Build **Neurons, Layers, and Multi-Layer Perceptrons** on top of that engine
- Train a tiny neural network to convergence

It is **intentionally minimal**. Every line is there to illuminate a concept, not to be production-ready. Think of it as a dissection of a larger framework, laid out on a table so you can see exactly how each organ works.

---

## 💡 Why I Built This

Modern deep learning libraries like PyTorch and TensorFlow are extraordinarily powerful — and extraordinarily opaque. When you call `loss.backward()`, thousands of lines of C++ CUDA code execute. That's great for productivity, but terrible for understanding.

I wanted to **break the black box**.

Specific questions I set out to answer by building this:

1. **What is a gradient, really?** Not the calculus definition — but what does it *mean* in the context of a neural network, and how does code compute it?
2. **How does the chain rule work in software?** It's one thing to apply it to a formula on paper; it's another to implement it for an arbitrary expression your user might write.
3. **How does PyTorch "know" which operations happened?** The answer turns out to be elegant: every arithmetic operation you do is secretly building a graph.
4. **Where do `weights` and `biases` come from?** What exactly is a "neuron" from a code perspective?
5. **Why do we zero gradients before each step?** (Spoiler: because they accumulate, and that's usually wrong.)

If you've ever used a deep learning framework and felt like you were pressing buttons without understanding the machine, this project is for you too.

---

## 🎓 What I Learned

### Automatic Differentiation is Just Graph Traversal
The "magic" of autograd is really just: (1) secretly recording every operation as a node in a graph, then (2) traversing that graph in reverse. There's no symbolic differentiation, no numerical approximation — just the chain rule applied mechanically to each node's local gradient formula.

### Topological Sorting is Non-Negotiable
You can't just walk the graph in any order. If node `B` depends on node `A`, you must compute `A`'s contribution to the gradient *before* you try to propagate through `A`. This requires a topological sort — the same algorithm used in build systems and package managers.

### Closures Make Backprop Elegant
Each operation's backward function is stored as a Python closure — a function that "closes over" the operands and the output, capturing them by reference. When called later, the closure knows exactly which inputs to update and by how much. This is how PyTorch stores gradient functions internally.

### The `+=` in Grad Accumulation is Critical
When a `Value` node is used in multiple places in the graph (e.g., a weight shared across neurons), its gradient gets contributions from *every* path. Using `+=` instead of `=` ensures all contributions accumulate correctly. Using `=` would be a subtle bug that silently gives wrong gradients.

### Object-Oriented Design Mirrors the Math
The `Module → Neuron → Layer → MLP` hierarchy isn't arbitrary — it exactly mirrors the mathematical structure of neural networks. The `parameters()` and `zero_grad()` interface mirrors what PyTorch's `nn.Module` does, because the problem demands it.

### Gradient Descent is Devastatingly Simple
After all the machinery, the training update is just:
```
parameter.data -= learning_rate * parameter.grad
```
Everything else — the graph, the topology, the chain rule — exists just to compute that `parameter.grad` correctly.

---

## ⚡ Quickstart

```bash
# Install in editable mode (no PyPI release needed)
pip install -e .
```

```python
from nanograd import Value, MLP

# --- Scalar autograd ---
x = Value(2.0)
y = Value(3.0)
z = x * y + x ** 2   # z = 2*3 + 2^2 = 10

z.backward()
print(x.grad)  # dz/dx = y + 2*x = 3 + 4 = 7.0
print(y.grad)  # dz/dy = x = 2.0

# --- Train a tiny MLP ---
model = MLP(3, [4, 4, 1])

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5],
      [0.5, 1.0,  1.0], [1.0,  1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]   # target labels

for step in range(100):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Gradient descent update
    for p in model.parameters():
        p.data -= 0.05 * p.grad

    if step % 10 == 0:
        print(f"step {step:3d} | loss {loss.data:.6f}")
```

---

## 📐 Theory: From Basic to Mastery

This section is a deep-dive into the concepts that power `nanograd`. Reading this alongside the source code in [`nanograd/engine.py`](nanograd/engine.py) will give you a complete picture.

---

### The Basics: Forward Passes & Derivatives

#### What is a Derivative?

A derivative tells you how much the output of a function changes if you nudge one of its inputs by a tiny amount. Formally:

$$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

In the context of a neural network, the *function* is the loss — a single scalar that measures how wrong our predictions are. The *inputs* we care about are the weights and biases. The derivative of the loss with respect to a weight tells us: *"if I increase this weight by a tiny bit, does the loss go up or down?"*

#### The Forward Pass

The forward pass is just ordinary arithmetic. Given inputs, compute the output. In `nanograd`, every scalar is wrapped in a `Value` object:

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c    # d.data = 2 * (-3) + 10 = 4.0
```

What makes this special is that each operation *records* its inputs. When we compute `a * b`, the result knows that it came from a multiplication of `a` and `b`. This record-keeping is what enables the backward pass.

#### What Derivatives Mean for a Network

Consider a single neuron's output:

$$o = \tanh(x_1 w_1 + x_2 w_2 + b)$$

- $x_1, x_2$: input features (fixed data)
- $w_1, w_2$: weights (parameters we learn)
- $b$: bias (also a parameter we learn)

After computing a loss $L$ based on this output, we want to know $\partial L / \partial w_1$ — "if I increase $w_1$, does the loss go up or down?" That tells us which direction to nudge the weight.

---

### Intermediate: The Chain Rule & Computational Graphs

#### The Computational Graph

Every expression built from `Value` objects creates a **directed acyclic graph (DAG)**. Each node is a `Value`; each edge points from an input to the operation that consumed it.

For the expression `L = (a * b + c) * d`:

```
a ──┐
    ├─[*]─> e ──┐
b ──┘           ├─[+]─> f ──┐
                             ├─[*]─> L
c ──────────────┘           │
                            d ──┘
```

The graph is built automatically as you write Python code. This is called **dynamic graph construction** (as opposed to TensorFlow 1.x's static graphs).

#### The Chain Rule

The chain rule from calculus says: if $L = f(g(x))$, then:

$$\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dx}$$

In the graph picture: if you want the gradient of $L$ with respect to a leaf node $x$, you multiply the gradients along every path from $x$ to $L$.

#### Local Gradients

Each operation knows its own **local gradient** — how much its output changes given a small change to each of its inputs, *holding everything else constant*. These are simple formulas:

| Operation | Output | Local grad w.r.t. `a` | Local grad w.r.t. `b` |
|-----------|--------|----------------------|----------------------|
| `a + b`   | `a + b` | `1` | `1` |
| `a * b`   | `a * b` | `b` | `a` |
| `a ** n`  | `aⁿ`    | `n * a^(n-1)` | — |
| `tanh(a)` | `tanh(a)` | `1 - tanh(a)²` | — |
| `exp(a)`  | `eᵃ` | `eᵃ` (= output) | — |

#### Applying the Chain Rule Backwards

The backward pass multiplies local gradients by the **upstream gradient** (the gradient flowing back from further down the computation chain). This is the core of backpropagation:

```
node.input.grad += local_gradient * node.grad
```

The `+=` accumulates contributions. The `* node.grad` applies the chain rule — multiplying by the upstream gradient.

In code, for multiplication:
```python
def _backward():
    self.grad  += other.data * out.grad   # d(self*other)/d(self) = other
    other.grad += self.data  * out.grad   # d(self*other)/d(other) = self
```

---

### Mastery: Topological Sort, Neurons & Gradient Descent

#### Topological Sorting for Backpropagation

We can't run the backward closures in any arbitrary order. Consider:

```
x ──> [A] ──> [B] ──> loss
          └──> [C] ──┘
```

Node `A` contributes to the loss through *two* paths (via `B` and via `C`). We must process `B` and `C` before we process `A`, so that `A`'s gradient accumulates contributions from both paths before `A`'s own backward function distributes it further.

A topological sort guarantees this. It orders nodes such that every node comes *before* all nodes that depend on it. By processing in **reverse** topological order (from the loss back to the inputs), we ensure every node's gradient is fully accumulated before we use it.

The implementation in `nanograd` uses a recursive depth-first search:

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)   # visit children first
        topo.append(v)           # append self AFTER children → post-order
```

Post-order DFS gives exactly the topological sort we need. Reversing it gives us the order to run backward closures.

#### Artificial Neurons in Code

An artificial neuron is a very simple structure:

$$\text{output} = \text{activation}\left(\sum_i w_i x_i + b\right)$$

In `nanograd.nn.Neuron`:

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
```

The weights start random (breaking symmetry so neurons learn different features). The bias starts at zero. The `tanh` activation squashes the output to `(-1, 1)`, which is useful for many tasks.

#### Layers and MLPs

A **Layer** is just a list of Neurons applied to the same input in parallel:
```python
class Layer:
    def __call__(self, x):
        return [n(x) for n in self.neurons]
```

A **Multi-Layer Perceptron (MLP)** stacks layers sequentially — the output of each layer becomes the input to the next:
```python
class MLP:
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

This creates deep representations: early layers detect simple patterns, later layers combine them into complex ones.

#### Gradient Descent and the Training Loop

Once we have gradients for all parameters, updating them is a one-liner:

```python
p.data -= learning_rate * p.grad
```

If `p.grad` is positive, the loss increases when `p` increases → we decrease `p`.  
If `p.grad` is negative, the loss decreases when `p` increases → we increase `p`.

The full training loop:

```python
for step in range(n_steps):
    # 1. Forward pass: compute loss
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 2. Backward pass: compute all gradients
    model.zero_grad()   # ← CRUCIAL: reset grads from last step
    loss.backward()

    # 3. Update: nudge every parameter in the right direction
    for p in model.parameters():
        p.data -= 0.05 * p.grad
```

Why `zero_grad()` before `backward()`? Because `grad` values are **accumulated** with `+=`. If you don't reset them, gradients from the previous step persist and corrupt the current update.

---

## 📁 Package Structure

```
nanograd/
├── nanograd/
│   ├── __init__.py     # Public API: Value, Neuron, Layer, MLP
│   ├── engine.py       # The autograd engine: Value class + all operations
│   └── nn.py           # Neural network modules: Module, Neuron, Layer, MLP
├── notebooks/
│   ├── nanograd.ipynb      # My own implementation (main learning notebook)
│   ├── nanograd_1.ipynb    # Reference notebook — early draft
│   └── nanograd_2.ipynb    # Reference notebook — complete version
├── tests/
│   └── test_engine.py  # Pytest suite: forward, backward, accumulation, MLP
├── setup.py            # Package installation config
└── README.md           # You are here
```

---

## 🧪 Running the Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests with verbose output
pytest tests/ -v

# Run a specific test class
pytest tests/test_engine.py::TestBackwardPass -v
```

Expected output:
```
tests/test_engine.py::TestForwardPass::test_addition PASSED
tests/test_engine.py::TestForwardPass::test_tanh PASSED
...
tests/test_engine.py::TestBackwardPass::test_complex_expression_matches_pytorch PASSED
tests/test_engine.py::TestMLP::test_loss_decreases PASSED

============ 21 passed in 0.12s ============
```

---

## 📓 Notebooks

The `notebooks/` directory preserves the original learning journey as Jupyter notebooks:

| Notebook | Contents |
|----------|----------|
| [`nanograd.ipynb`](notebooks/nanograd.ipynb) | My primary implementation — follows the tutorial step-by-step with my own annotations |
| [`nanograd_1.ipynb`](notebooks/nanograd_1.ipynb) | Reference: early draft from Karpathy's tutorial (basic Value class + backprop) |
| [`nanograd_2.ipynb`](notebooks/nanograd_2.ipynb) | Reference: complete version with neural network classes and training loop |

---

## 🙏 Acknowledgements

- **[Andrej Karpathy](https://github.com/karpathy)** — whose [micrograd](https://github.com/karpathy/micrograd) library and accompanying YouTube tutorial inspired every line of this project. The tutorial is freely available and is one of the best educational resources on neural networks ever created.
- **[PyTorch](https://pytorch.org/)** — whose design informed the `Module/parameters/zero_grad` API of `nanograd.nn`.

---

*"The best way to understand something is to build it." — folk wisdom among engineers*
