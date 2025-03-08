def step_function(x):
    return 1 if x > 0 else 0

def neuron(input, weight, bias):
    weighted_sum = sum(w * i for w, i in zip(weight, input))
    return step_function(weighted_sum + bias)

# Example usage when run directly
if __name__ == "__main__":
    # Example with the original values
    result = neuron([1, 0.1], [0.7, 0.5], -0.7)
    print(f"Neuron output: {result}")

