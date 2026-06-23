# Backpropagation Implementation

This readme contains an implementation of the Multi Layer Observation and backpropagation algorithm as described in "Learning representations by back-propagating errors" (Rumelhart, Hinton, & Williams, 1986).

## Overview
This project tackles the **Symmetry Task** outlined in the paper. The network takes a 6-bit binary array and learns to classify whether the array is symmetrical around its center. 

* **Note:** This implementation builds the forward pass, backward pass, and weight updates completely from scratch using standard Python/PyTorch tensors, without leaning on `torch.autograd`.

## Dependencies
* Python 3.8+
* PyTorch
* Numpy

## How to Run
To train the network and view the evaluation logs, run:
`python src/code.py`

## Expected Results
When running the script, the network trains for 30000 sweeps. 
* **Initial Error:** approx 0.25
* **Final Error:** 0.0000
The final output activations for symmetrical inputs consistently score above 0.8, and asymmetrical inputs score below 0.2
