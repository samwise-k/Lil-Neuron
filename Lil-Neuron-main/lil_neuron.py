def step_function(x):
    return 1 if x > 0 else 0

def neuron(input, weight, bias):
    weighted_sum = sum(w * i for w, i in zip(weight, input))
    return step_function(weighted_sum + bias)

def multilayer_network(inputs, weights_hidden, biases_hidden, weights_output, bias_output):
    # Hidden layer computation
    hidden_outputs = []
    for w, b in zip(weights_hidden, biases_hidden):
        hidden_output = neuron(inputs, w, b)
        hidden_outputs.append(hidden_output)
    
    # Output layer computation
    output = neuron(hidden_outputs, weights_output, bias_output)
    return hidden_outputs, output

if __name__ == "__main__":
    # Example for single neuron
    result = neuron([1, 0.1], [0.7, 0.5], -0.7)
    print(f"Neuron output: {result}")
    
    # Example for multi-layer network
    inputs = [1, 0.1]
    weights_hidden = [[0.7, 0.5], [0.3, -0.2]]
    biases_hidden = [-0.7, 0.1]
    weights_output = [0.4, -0.6]
    bias_output = -0.3
    hidden_outputs, output = multilayer_network(inputs, weights_hidden, biases_hidden, weights_output, bias_output)
    print(f"Hidden Layer Outputs: {hidden_outputs}")
    print(f"Final Output: {output}")