import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from lil_neuron import step_function, neuron, multilayer_network

st.set_page_config(page_title="Neuron Visualization Dashboard", layout="wide")

st.title("Interactive Neuron Visualization")
st.markdown("""
This dashboard visualizes how artificial neurons process inputs. Choose between a single neuron or a multi-layer network.
""")

# Use radio buttons to select which visualization to display
selected_tab = st.radio("Select visualization:", ["Single Neuron", "Multi-Layer Network"], horizontal=True, label_visibility="collapsed")

# Sidebar parameters based on selected tab
if selected_tab == "Single Neuron":
    # Single Neuron Parameters
    st.sidebar.header("Single Neuron Parameters")
    input_1 = st.sidebar.slider("Input 1", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="single_input_1")
    input_2 = st.sidebar.slider("Input 2", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="single_input_2")
    weight_1 = st.sidebar.slider("Weight 1", min_value=-1.0, max_value=1.0, value=0.7, step=0.1, key="single_weight_1")
    weight_2 = st.sidebar.slider("Weight 2", min_value=-1.0, max_value=1.0, value=0.5, step=0.1, key="single_weight_2")
    bias = st.sidebar.slider("Bias", min_value=-1.0, max_value=1.0, value=-0.7, step=0.1, key="single_bias")
else:
    # Multi-Layer Network Parameters
    st.sidebar.header("Multi-Layer Network Parameters")
    ml_input_1 = st.sidebar.slider("Input 1", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="ml_input_1")
    ml_input_2 = st.sidebar.slider("Input 2", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="ml_input_2")
    st.sidebar.subheader("Hidden Layer Weights")
    w_h1_i1 = st.sidebar.slider("W Hidden1-Input1", min_value=-1.0, max_value=1.0, value=0.7, step=0.1)
    w_h1_i2 = st.sidebar.slider("W Hidden1-Input2", min_value=-1.0, max_value=1.0, value=0.5, step=0.1)
    w_h2_i1 = st.sidebar.slider("W Hidden2-Input1", min_value=-1.0, max_value=1.0, value=0.3, step=0.1)
    w_h2_i2 = st.sidebar.slider("W Hidden2-Input2", min_value=-1.0, max_value=1.0, value=-0.2, step=0.1)
    st.sidebar.subheader("Hidden Layer Biases")
    b_h1 = st.sidebar.slider("Bias Hidden1", min_value=-1.0, max_value=1.0, value=-0.7, step=0.1)
    b_h2 = st.sidebar.slider("Bias Hidden2", min_value=-1.0, max_value=1.0, value=0.1, step=0.1)
    st.sidebar.subheader("Output Layer Weights")
    w_o_h1 = st.sidebar.slider("W Output-Hidden1", min_value=-1.0, max_value=1.0, value=0.4, step=0.1)
    w_o_h2 = st.sidebar.slider("W Output-Hidden2", min_value=-1.0, max_value=1.0, value=-0.6, step=0.1)
    st.sidebar.subheader("Output Layer Bias")
    b_o = st.sidebar.slider("Bias Output", min_value=-1.0, max_value=1.0, value=-0.3, step=0.1)

# --- Single Neuron Visualization ---
if selected_tab == "Single Neuron":
    st.header("Single Neuron Visualization")
    st.markdown("""
    This section visualizes how a simple artificial neuron processes inputs, applies weights and bias, 
    and produces an output using a step activation function.
    """)

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

    with col2:
        st.header("Neuron Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter([0, 0], [1, 0], s=300, color='blue', zorder=5)
        ax.text(-0.15, 1, f"{input_1}", fontsize=12, ha='right')
        ax.text(-0.15, 0, f"{input_2}", fontsize=12, ha='right')
        ax.scatter([1], [0.5], s=500, color='green' if output == 1 else 'red', zorder=5)
        ax.text(1, 0.5, f"{output}", fontsize=15, ha='center', va='center', color='white')
        ax.plot([0, 1], [1, 0.5], 'k-', linewidth=2 * abs(weight_1))
        ax.plot([0, 1], [0, 0.5], 'k-', linewidth=2 * abs(weight_2))
        ax.text(0.4, 0.85, f"w1: {weight_1}", fontsize=10)
        ax.text(0.4, 0.15, f"w2: {weight_2}", fontsize=10)
        ax.text(1.1, 0.5, f"bias: {bias}", fontsize=10, ha='left')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title("Neuron Architecture")
        st.pyplot(fig)

    # Step Activation Function Visualization
    st.subheader("Step Activation Function")
    x = np.linspace(-2, 2, 1000)
    y = [step_function(val) for val in x]
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(x, y, 'b-', linewidth=2)
    ax2.axvline(x=0, color='gray', linestyle='--')
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.axhline(y=1, color='gray', linestyle='--')
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

    # Decision Boundary Visualization
    st.header("Decision Boundary Visualization")
    st.write("""
    The decision boundary shows when the neuron will output 1 (green region) vs 0 (red region)
    based on different combinations of input values.
    """)
    x1_range = np.linspace(0, 1, 100)
    x2_range = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            Z[j, i] = neuron([X1[j, i], X2[j, i]], weights, bias)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    contour = ax3.contourf(X1, X2, Z, levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.5)
    ax3.set_xlabel("Input 1")
    ax3.set_ylabel("Input 2")
    ax3.set_title("Decision Boundary")
    cbar = plt.colorbar(contour, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['0', '1'])
    ax3.scatter([input_1], [input_2], color='blue', s=200, edgecolor='black', zorder=5)
    ax3.text(input_1 + 0.05, input_2, f"Current Input ({input_1}, {input_2})", fontsize=10)
    if weight_2 != 0:
        x1 = np.linspace(0, 1, 100)
        x2 = (-weight_1 * x1 - bias) / weight_2
        valid_idx = (x2 >= 0) & (x2 <= 1)
        ax3.plot(x1[valid_idx], x2[valid_idx], 'k--', linewidth=2)
        ax3.text(0.5, 0.5, f"{weight_1}*x1 + {weight_2}*x2 + {bias} = 0", 
                 bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    st.pyplot(fig3)

# --- Multi-Layer Network Visualization ---
else:
    st.header("Multi-Layer Network Visualization")
    st.markdown("""
    This section visualizes a multi-layer network with 2 input neurons, 2 hidden neurons, and 1 output neuron.
    """)

    # Compute multi-layer network
    inputs = [ml_input_1, ml_input_2]
    weights_hidden = [[w_h1_i1, w_h1_i2], [w_h2_i1, w_h2_i2]]
    biases_hidden = [b_h1, b_h2]
    weights_output = [w_o_h1, w_o_h2]
    bias_output = b_o
    
    hidden_outputs, output = multilayer_network(inputs, weights_hidden, biases_hidden, weights_output, bias_output)
    
    # Calculate activations for visualization
    h1_activation = inputs[0] * w_h1_i1 + inputs[1] * w_h1_i2 + b_h1
    h2_activation = inputs[0] * w_h2_i1 + inputs[1] * w_h2_i2 + b_h2
    o_activation = hidden_outputs[0] * w_o_h1 + hidden_outputs[1] * w_o_h2 + b_o

    # Display calculations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Calculation Steps")
        st.write("**Hidden Layer:**")
        st.write(f"H1: ({inputs[0]} × {w_h1_i1}) + ({inputs[1]} × {w_h1_i2}) + {b_h1} = {h1_activation} → {hidden_outputs[0]}")
        st.write(f"H2: ({inputs[0]} × {w_h2_i1}) + ({inputs[1]} × {w_h2_i2}) + {b_h2} = {h2_activation} → {hidden_outputs[1]}")
        st.write("**Output Layer:**")
        st.write(f"Output: ({hidden_outputs[0]} × {w_o_h1}) + ({hidden_outputs[1]} × {w_o_h2}) + {b_o} = {o_activation} → {output}")

    with col2:
        st.subheader("Network Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter([0, 0], [1, 0], s=300, color='blue', zorder=5)
        ax.text(-0.15, 1, f"{inputs[0]}", fontsize=12, ha='right')
        ax.text(-0.15, 0, f"{inputs[1]}", fontsize=12, ha='right')
        ax.scatter([1, 1], [1, 0], s=400, color=['green' if h == 1 else 'red' for h in hidden_outputs], zorder=5)
        ax.text(1, 1, f"{hidden_outputs[0]}", fontsize=12, ha='center', va='center', color='white')
        ax.text(1, 0, f"{hidden_outputs[1]}", fontsize=12, ha='center', va='center', color='white')
        ax.scatter([2], [0.5], s=500, color='green' if output == 1 else 'red', zorder=5)
        ax.text(2, 0.5, f"{output}", fontsize=15, ha='center', va='center', color='white')
        ax.plot([0, 1], [1, 1], 'k-', linewidth=2 * abs(w_h1_i1))
        ax.plot([0, 1], [0, 1], 'k-', linewidth=2 * abs(w_h1_i2))
        ax.plot([0, 1], [1, 0], 'k-', linewidth=2 * abs(w_h2_i1))
        ax.plot([0, 1], [0, 0], 'k-', linewidth=2 * abs(w_h2_i2))
        ax.plot([1, 2], [1, 0.5], 'k-', linewidth=2 * abs(w_o_h1))
        ax.plot([1, 2], [0, 0.5], 'k-', linewidth=2 * abs(w_o_h2))
        ax.text(0.4, 1.1, f"{w_h1_i1}", fontsize=10)
        ax.text(0.4, 0.8, f"{w_h1_i2}", fontsize=10)
        ax.text(0.4, 0.2, f"{w_h2_i1}", fontsize=10)
        ax.text(0.4, -0.1, f"{w_h2_i2}", fontsize=10)
        ax.text(1.4, 0.85, f"{w_o_h1}", fontsize=10)
        ax.text(1.4, 0.15, f"{w_o_h2}", fontsize=10)
        ax.text(1.1, 1, f"b: {b_h1}", fontsize=10, ha='left')
        ax.text(1.1, 0, f"b: {b_h2}", fontsize=10, ha='left')
        ax.text(2.1, 0.5, f"b: {b_o}", fontsize=10, ha='left')
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Multi-Layer Network")
        st.pyplot(fig)

    # Step Activation Function Visualization for Multi-Layer
    st.subheader("Step Activation Functions")
    x = np.linspace(-2, 2, 1000)
    y = [step_function(val) for val in x]
    
    # H1 Activation Plot
    fig_h1, ax_h1 = plt.subplots(figsize=(10, 3))
    ax_h1.plot(x, y, 'b-', linewidth=2)
    ax_h1.axvline(x=0, color='gray', linestyle='--')
    ax_h1.axhline(y=0, color='gray', linestyle='--')
    ax_h1.axhline(y=1, color='gray', linestyle='--')
    ax_h1.scatter([h1_activation], [hidden_outputs[0]], color='red', s=100, zorder=5)
    ax_h1.annotate(f"({h1_activation:.2f}, {hidden_outputs[0]})", 
                   xy=(h1_activation, hidden_outputs[0]), 
                   xytext=(h1_activation + 0.3, hidden_outputs[0] + 0.3),
                   arrowprops=dict(facecolor='black', shrink=0.05))
    ax_h1.set_xlabel("Input to H1")
    ax_h1.set_ylabel("Output")
    ax_h1.set_title("H1 Step Function")
    ax_h1.grid(True)
    st.pyplot(fig_h1)

    # H2 Activation Plot
    fig_h2, ax_h2 = plt.subplots(figsize=(10, 3))
    ax_h2.plot(x, y, 'b-', linewidth=2)
    ax_h2.axvline(x=0, color='gray', linestyle='--')
    ax_h2.axhline(y=0, color='gray', linestyle='--')
    ax_h2.axhline(y=1, color='gray', linestyle='--')
    ax_h2.scatter([h2_activation], [hidden_outputs[1]], color='red', s=100, zorder=5)
    ax_h2.annotate(f"({h2_activation:.2f}, {hidden_outputs[1]})", 
                   xy=(h2_activation, hidden_outputs[1]), 
                   xytext=(h2_activation + 0.3, hidden_outputs[1] + 0.3),
                   arrowprops=dict(facecolor='black', shrink=0.05))
    ax_h2.set_xlabel("Input to H2")
    ax_h2.set_ylabel("Output")
    ax_h2.set_title("H2 Step Function")
    ax_h2.grid(True)
    st.pyplot(fig_h2)

    # Output Activation Plot
    fig_o, ax_o = plt.subplots(figsize=(10, 3))
    ax_o.plot(x, y, 'b-', linewidth=2)
    ax_o.axvline(x=0, color='gray', linestyle='--')
    ax_o.axhline(y=0, color='gray', linestyle='--')
    ax_o.axhline(y=1, color='gray', linestyle='--')
    ax_o.scatter([o_activation], [output], color='red', s=100, zorder=5)
    ax_o.annotate(f"({o_activation:.2f}, {output})", 
                  xy=(o_activation, output), 
                  xytext=(o_activation + 0.3, output + 0.3),
                  arrowprops=dict(facecolor='black', shrink=0.05))
    ax_o.set_xlabel("Input to Output")
    ax_o.set_ylabel("Output")
    ax_o.set_title("Output Step Function")
    ax_o.grid(True)
    st.pyplot(fig_o)

    # Decision Boundary Visualization for Multi-Layer
    st.header("Decision Boundary Visualization")
    st.write("""
    The decision boundary shows when the network outputs 1 (green region) vs 0 (red region)
    based on different combinations of input values.
    """)
    x1_range = np.linspace(0, 1, 100)
    x2_range = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            _, z = multilayer_network([X1[j, i], X2[j, i]], weights_hidden, biases_hidden, weights_output, bias_output)
            Z[j, i] = z
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    contour = ax4.contourf(X1, X2, Z, levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.5)
    ax4.set_xlabel("Input 1")
    ax4.set_ylabel("Input 2")
    ax4.set_title("Decision Boundary")
    cbar = plt.colorbar(contour, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['0', '1'])
    ax4.scatter([ml_input_1], [ml_input_2], color='blue', s=200, edgecolor='black', zorder=5)
    ax4.text(ml_input_1 + 0.05, ml_input_2, f"Current Input ({ml_input_1}, {ml_input_2})", fontsize=10)
    st.pyplot(fig4)

# Sidebar instructions (always visible)
st.sidebar.markdown("---")
st.sidebar.header("How to Use")
st.sidebar.write("""
1. Switch between visualizations using the radio buttons at the top.
2. Adjust sliders to change inputs, weights, and biases.
3. Observe how changes affect the outputs and visualizations.
""")