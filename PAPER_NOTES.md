# Paper Notes: Rumelhart et al. (1986) - Backpropagation

### 1. What's the central claim?
The main argument here is that the "back-propagation" method finally gives us a way to train hidden layers in neural nets. Before this, simple perceptrons couldn't handle complex, non-linear problems because no one knew how to update weights for hidden nodes (they couldn't learn "internal representations"). The authors claim that by using the chain rule to pass errors backward from the output layer, the network can actually figure out what features to look for and solve these harder problems.

### 2. What needs to be implemented?
I need to build a standard feed-forward Multi-Layer Perceptron (MLP) from scratch. Since I can't just call `loss.backward()` in PyTorch for this challenge, the core thing to implement is the manual math for the forward and backward passes.

**The Math:**
* **Forward pass:** * Calculate total input: $x_j = \sum y_i w_{ji}$
    * Apply the Sigmoid activation: $y_j = 1 / (1 + e^{-x_j})$
* **Loss:** Mean Squared Error (MSE).
* **Backward pass (the hard part):** * Output error is actual minus desired: $(y - d)$
    * Sigmoid derivative is: $y(1 - y)$
    * Apply the chain rule to push these gradients back to the hidden layer weights.
* **Weight updates:** Standard gradient descent, but the paper uses a momentum term (they call it $lpha$) to speed up convergence.

### 3. Dataset, Metrics, and Baseline
* **Dataset:** I'm going with the "Symmetry Task" from Figure 1. The input is a random 6-bit array (like `1 0 1 1 0 1`), and the target is `1` if it's perfectly symmetrical, and `0` otherwise.
* **Metrics:** The main metric is the Total Error (MSE) over the training sweeps. The goal is to see it drop to near zero. The paper also mentions checking if the output unit activations map cleanly to the targets—specifically, checking if True is > 0.8 and False is < 0.2.
* **Baseline:** The implied baseline is a standard single-layer perceptron. A normal perceptron completely fails at the symmetry task because detecting symmetry requires comparing inputs together, which strictly requires a hidden layer. My network just needs to successfully converge where those older methods hit a wall.
