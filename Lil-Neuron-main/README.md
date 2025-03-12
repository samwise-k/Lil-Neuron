# Neuron Visualization Dashboard
Now available at https://lil-neurongit-v1.streamlit.app/
This project provides an interactive dashboard to visualize how artificial neurons process inputs. It includes both a simple single neuron visualization and a multi-layer network visualization.

## Features

- Two visualization modes:
  - Single Neuron: Visualize how a simple artificial neuron processes inputs
  - Multi-Layer Network: Visualize how a network with hidden layers processes inputs
- Interactive sliders to adjust input values, weights, and biases
- Real-time visualization of neuron calculations
- Visual representation of neuron architecture
- Step function visualization
- Decision boundary visualization
- Detailed explanation of each step in the computation

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

1. Select the visualization mode (Single Neuron or Multi-Layer Network)

2. For Single Neuron mode:
   - Use the sliders in the sidebar to adjust:
     - Input values (Input 1 and Input 2)
     - Weight values (Weight 1 and Weight 2)
     - Bias value

3. For Multi-Layer Network mode:
   - Use the sliders in the sidebar to adjust:
     - Input values (Input 1 and Input 2)
     - Hidden layer weights and biases
     - Output layer weights and bias

4. Observe how these changes affect:
   - The calculation steps
   - The network's output
   - The visualization of the neuron architecture
   - The step function and where the current activation falls
   - The decision boundary in the input space

## Understanding the Models

### Single Neuron
The artificial neuron:
1. Takes two input values
2. Multiplies each input by its corresponding weight
3. Sums these weighted inputs
4. Adds a bias term
5. Applies a step activation function (outputs 1 if the result is > 0, otherwise 0)

### Multi-Layer Network
The multi-layer network:
1. Takes two input values
2. Processes them through a hidden layer with two neurons
3. Each hidden neuron applies weights, bias, and a step activation function
4. The output neuron takes the outputs from the hidden layer as inputs
5. Applies its own weights, bias, and activation function to produce the final output

The decision boundary visualizations show all possible combinations of inputs that would cause the neuron/network to fire (output 1).

## Files

- `lil_neuron.py`: Contains the core neuron and multi-layer network implementation
- `neuron_dashboard.py`: Streamlit dashboard for visualization
- `requirements.txt`: Required Python packages 
