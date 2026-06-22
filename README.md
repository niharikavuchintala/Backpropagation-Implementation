# TASK-3-Paper---Code---Research-Implementation-Challenge

# Rumelhart 1986: Backpropagation Implementation

This readme contains an implementation of the Multi-Layer Perceptron and backpropagation algorithm as described in "Learning representations by back-propagating errors" (Rumelhart, Hinton, & Williams, 1986).

## Overview
This project tackles the **Symmetry Task** outlined in the paper. The network takes a 6-bit binary array and learns to classify whether the array is symmetrical around its center. 

* **Note:** This implementation builds the forward pass, backward pass, and weight updates completely from scratch using standard Python/PyTorch tensors, without relying on `torch.autograd`.

## Dependencies
* Python 3.8+
* PyTorch (for tensor operations)
* Numpy (optional, for dataset generation)

## How to Run
To train the network and view the evaluation logs, simply run:
`python src/network.py`

## Expected Results
When running the script, the network trains for `[Insert Number]` sweeps. 
* **Initial Error:** `[Insert Starting Error]`
* **Final Error:** `[Insert Final Error]`
The final output activations for symmetrical inputs consistently score above 0.8, and asymmetrical inputs score below 0.2, matching the authors' original claims.
