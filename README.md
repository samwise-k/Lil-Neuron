# Neuron Visualization Dashboard

This project provides an interactive dashboard to visualize how a simple artificial neuron processes inputs, applies weights and bias, and produces an output using a step activation function.

## Features

- Interactive sliders to adjust input values, weights, and bias
- Real-time visualization of neuron calculations
- Visual representation of the neuron architecture
- Step function visualization
- Decision boundary visualization
- Detailed explanation of each step in the neuron's computation

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit dashboard:

```bash
streamlit run neuron_dashboard.py
```

This will open a web browser with the interactive dashboard.

## How to Use the Dashboard

1. Use the sliders in the sidebar to adjust:
   - Input values (Input 1 and Input 2)
   - Weight values (Weight 1 and Weight 2)
   - Bias value

2. Observe how these changes affect:
   - The neuron's calculation steps
   - The neuron's output (0 or 1)
   - The visualization of the neuron architecture
   - The step function and where the current activation falls
   - The decision boundary in the input space

## Understanding the Neuron

The artificial neuron in this dashboard:

1. Takes two input values
2. Multiplies each input by its corresponding weight
3. Sums these weighted inputs
4. Adds a bias term
5. Applies a step activation function (outputs 1 if the result is > 0, otherwise 0)

The decision boundary visualization shows all possible combinations of inputs that would cause the neuron to fire (output 1).

## Files

- `lil_neuron.py`: Contains the core neuron implementation
- `neuron_dashboard.py`: Streamlit dashboard for visualization
- `requirements.txt`: Required Python packages 