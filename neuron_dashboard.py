import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from lil_neuron import step_function, neuron

st.set_page_config(page_title="Neuron Visualization Dashboard", layout="wide")

st.title("Interactive Neuron Visualization")
st.markdown("""
This dashboard visualizes how a simple artificial neuron processes inputs, applies weights and bias, 
and produces an output using a step activation function.
""")

# Sidebar for input parameters
st.sidebar.header("Neuron Parameters")

# Input values
st.sidebar.subheader("Input Values")
input_1 = st.sidebar.slider("Input 1", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
input_2 = st.sidebar.slider("Input 2", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

# Weight values
st.sidebar.subheader("Weight Values")
weight_1 = st.sidebar.slider("Weight 1", min_value=-1.0, max_value=1.0, value=0.7, step=0.1)
weight_2 = st.sidebar.slider("Weight 2", min_value=-1.0, max_value=1.0, value=0.5, step=0.1)

# Bias value
st.sidebar.subheader("Bias Value")
bias = st.sidebar.slider("Bias", min_value=-1.0, max_value=1.0, value=-0.7, step=0.1)

# Calculate neuron output
inputs = [input_1, input_2]
weights = [weight_1, weight_2]
weighted_sum = sum(w * i for w, i in zip(weights, inputs))
activation = weighted_sum + bias
output = neuron(inputs, weights, bias)

# Display the calculation steps
col1, col2 = st.columns(2)

with col1:
    st.header("Neuron Calculation Steps")
    
    st.subheader("Step 1: Inputs and Weights")
    st.write(f"Input 1: {input_1} × Weight 1: {weight_1} = {input_1 * weight_1}")
    st.write(f"Input 2: {input_2} × Weight 2: {weight_2} = {input_2 * weight_2}")
    
    st.subheader("Step 2: Calculate Weighted Sum")
    st.write(f"Weighted Sum = {input_1 * weight_1} + {input_2 * weight_2} = {weighted_sum}")
    
    st.subheader("Step 3: Add Bias")
    st.write(f"Weighted Sum + Bias = {weighted_sum} + {bias} = {activation}")
    
    st.subheader("Step 4: Apply Activation Function")
    st.write(f"Step Function({activation}) = {output}")
    
    st.subheader("Final Output")
    st.markdown(f"<h1 style='text-align: center; color: {'green' if output == 1 else 'red'};'>{output}</h1>", unsafe_allow_html=True)

# Visualization
with col2:
    st.header("Neuron Visualization")
    
    # Create a figure for the neuron visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw neuron components
    # Input nodes
    ax.scatter([0, 0], [1, 0], s=300, color='blue', zorder=5)
    ax.text(-0.15, 1, f"{input_1}", fontsize=12, ha='right')
    ax.text(-0.15, 0, f"{input_2}", fontsize=12, ha='right')
    
    # Neuron node
    ax.scatter([1], [0.5], s=500, color='green' if output == 1 else 'red', zorder=5)
    ax.text(1, 0.5, f"{output}", fontsize=15, ha='center', va='center', color='white')
    
    # Connection lines with weights
    ax.plot([0, 1], [1, 0.5], 'k-', linewidth=2 * abs(weight_1))
    ax.plot([0, 1], [0, 0.5], 'k-', linewidth=2 * abs(weight_2))
    
    # Weight labels
    ax.text(0.4, 0.85, f"w1: {weight_1}", fontsize=10)
    ax.text(0.4, 0.15, f"w2: {weight_2}", fontsize=10)
    
    # Bias label
    ax.text(1.1, 0.5, f"bias: {bias}", fontsize=10, ha='left')
    
    # Set axis limits and remove ticks
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title("Neuron Architecture")
    
    st.pyplot(fig)
    
    # Plot the step function
    st.subheader("Step Activation Function")
    x = np.linspace(-2, 2, 1000)
    y = [step_function(val) for val in x]
    
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(x, y, 'b-', linewidth=2)
    ax2.axvline(x=0, color='gray', linestyle='--')
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.axhline(y=1, color='gray', linestyle='--')
    
    # Mark the current activation value
    ax2.scatter([activation], [output], color='red', s=100, zorder=5)
    ax2.annotate(f"({activation:.2f}, {output})", 
                xy=(activation, output), 
                xytext=(activation + 0.3, output + 0.3),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax2.set_xlabel("Input")
    ax2.set_ylabel("Output")
    ax2.set_title("Step Function: Output = 1 if Input > 0 else 0")
    ax2.grid(True)
    
    st.pyplot(fig2)

# Decision boundary visualization
st.header("Decision Boundary Visualization")
st.write("""
The decision boundary shows when the neuron will output 1 (green region) vs 0 (red region)
based on different combinations of input values.
""")

# Create a grid of input values
x1_range = np.linspace(0, 1, 100)
x2_range = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Calculate the decision boundary
Z = np.zeros_like(X1)
for i in range(len(x1_range)):
    for j in range(len(x2_range)):
        Z[j, i] = neuron([X1[j, i], X2[j, i]], weights, bias)

# Plot the decision boundary
fig3, ax3 = plt.subplots(figsize=(10, 8))
contour = ax3.contourf(X1, X2, Z, levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.5)
ax3.set_xlabel("Input 1")
ax3.set_ylabel("Input 2")
ax3.set_title("Decision Boundary")

# Add a colorbar
cbar = plt.colorbar(contour, ticks=[0.25, 0.75])
cbar.ax.set_yticklabels(['0', '1'])

# Plot the current input point
ax3.scatter([input_1], [input_2], color='blue', s=200, edgecolor='black', zorder=5)
ax3.text(input_1 + 0.05, input_2, f"Current Input ({input_1}, {input_2})", fontsize=10)

# Draw the decision boundary line
# For a neuron with 2 inputs, the boundary is a line: w1*x1 + w2*x2 + bias = 0
if weight_2 != 0:
    x1 = np.linspace(0, 1, 100)
    x2 = (-weight_1 * x1 - bias) / weight_2
    valid_idx = (x2 >= 0) & (x2 <= 1)
    ax3.plot(x1[valid_idx], x2[valid_idx], 'k--', linewidth=2)
    ax3.text(0.5, 0.5, f"{weight_1}*x1 + {weight_2}*x2 + {bias} = 0", 
             bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

st.pyplot(fig3)

# Add explanation of the neuron
st.header("How This Neuron Works")
st.write("""
1. **Inputs**: The neuron receives two input values (between 0 and 1).
2. **Weights**: Each input is multiplied by its corresponding weight.
3. **Weighted Sum**: The products of inputs and weights are summed together.
4. **Bias**: A bias term is added to the weighted sum.
5. **Activation Function**: The step function is applied to the result:
   - If the result is greater than 0, the neuron outputs 1
   - Otherwise, it outputs 0

The decision boundary shows all possible combinations of inputs that would cause the neuron to fire (output 1).
""")

# Add instructions for using the dashboard
st.sidebar.markdown("---")
st.sidebar.header("How to Use")
st.sidebar.write("""
1. Adjust the sliders to change input values, weights, and bias
2. See how these changes affect the neuron's output
3. Observe the decision boundary changing based on weights and bias
""") 