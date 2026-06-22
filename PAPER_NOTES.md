# Paper Notes: Backpropagation

### 1. What's the central claim?
The main argument here is that the "back-propagation" method gives us a way to train hidden layers in neural nets finally. Before this, simple observations could not solve complex, non-linear problems because no one knew how to update the numbers of hidden nodes (they could not learn "internal representations"). The network can actually learn what features to look for and solve these harder problems by using the chain rule to rectify errors from the output layer, as said by the authors.

### 2. What needs to be implemented?
I have to build a basic feed forward MLP from scratch. Since we can’t just call `loss.backward()` in PyTorch for this problem, the key thing to get down are the equations for manually computing the forward pass and the backpropagation.

**Math Formulaes:**
* **Forward pass:** Calculate total input: $x_j = \sum y_i w_{ji}$
    * Apply the Sigmoid activation: $y_j = 1 / (1 + e^{-x_j})$
* **Loss:** Mean Squared Error (MSE).
* **Backward pass:** $(y - d)$
    * Sigmoid derivative is: $y(1 - y)$
    * Apply the chain rule to push these gradients back to the hidden layer weights.
* **Weight updates:** Standard gradient descent, but the paper uses a momentum term to speed up convergence.

### 3. Dataset, Metrics, and Baseline
* **Dataset:** I'm going with the "Symmetry Task" from Figure 1.
* The input is a random 6-bit array (like `1 0 1 1 0 1`), and the target is `1` if it's perfectly symmetrical, and `0` otherwise.
* **Metrics:** The main metric is the Total Error (MSE) over the training sweeps. The goal is to see it drop to near zero. The paper also mentions checking if the output unit activations map cleanly to the targets specifically, checking if True is > 0.8 and False is < 0.2.
* **Baseline:** The implied baseline is a standard single layer observation. A normal observation completely fails at the symmetry task because detecting symmetry requires comparing inputs together, which requires a hidden layer. The network needs to successfully meet where those older methods hit a wall.
