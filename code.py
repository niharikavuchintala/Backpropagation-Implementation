import torch
import itertools

def make_dataset():
    X_list = []
    d_list = []
    
    combinations = list(itertools.product([0.0, 1.0], repeat=6))
    
    for combo in combinations:

        combo_list = list(combo)
        
        if combo_list == combo_list[::-1]:
            label = 1.0  
        else:
            label = 0.0
            
        X_list.append(combo_list)
        d_list.append([label])
        
    X = torch.tensor(X_list)
    d = torch.tensor(d_list)
    
    return X, d

X, d = make_dataset()
print(f"Dataset generated! X shape: {X.shape}, d shape: {d.shape}")

W1 = torch.rand(6, 8) * 0.6 - 0.3
W2 = torch.rand(8, 1) * 0.6 - 0.3

epsilon = 0.5 

def sigmoid(x):

    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(y):

    return y * (1.0 - y)

epochs = 30000
print("\nTraining Starting...")
for epoch in range(epochs):

    hidden_input = torch.matmul(X, W1)
    hidden_output = sigmoid(hidden_input)
    
    final_input = torch.matmul(hidden_output, W2)
    predicted_output = sigmoid(final_input)

    output_error = d - predicted_output
    
    output_gradient = output_error * sigmoid_derivative(predicted_output)
    
    hidden_error = torch.matmul(output_gradient, W2.T)
    
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)
    
    W2 += epsilon * torch.matmul(hidden_output.T, output_gradient)
    W1 += epsilon * torch.matmul(X.T, hidden_gradient)
    
    if epoch % 1000 == 0:
        loss = torch.mean(torch.square(output_error))
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("\nFinal Tests")

print("Testing symmetric array: [1, 0, 1, 1, 0, 1]")
input_1 = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

hidden_1 = sigmoid(torch.matmul(input_1, W1))
answer_1 = sigmoid(torch.matmul(hidden_1, W2))
print("Network guess:", round(answer_1.item(), 4))

print("Testing asymmetric array: [1, 0, 0, 0, 0, 0]")
input_2 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

hidden_2 = sigmoid(torch.matmul(input_2, W1))
answer_2 = sigmoid(torch.matmul(hidden_2, W2))
print("Network guess:", round(answer_2.item(), 4))