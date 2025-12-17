import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import plotly.graph_objects as go
import pandas as pd
import html
import glob
import io
from model import MD4Config, MD4

st.set_page_config(layout="wide", page_title="Discrete Diffusion Viz")

@st.cache_resource
def get_resources():
    with open("blobs/wine_0.txt", "r", encoding="utf-8") as f0, \
        open("blobs/wine_1.txt", "r", encoding="utf-8") as f1, \
        open("blobs/wine_2.txt", "r", encoding="utf-8") as f2:
        text = f0.read() + f1.read() + f2.read()
    # Create train / val tensor
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Total Text length: {len(text)} characters")
    print(f"Vocab size: {vocab_size}")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    config = MD4Config(len(chars))
    return config, stoi, itos

@st.cache_resource
def load_checkpoint_model(device_name):
    """Loads the model and weights."""
    # Initialize architecture
    model = MD4(config)

    def load_chunked(file_prefix, map_location=None):
        """
        Loads parts (file_prefix.part000, .part001, ...) back into an object.
        """
        pattern = f"{file_prefix}.part*"
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No chunk files found for prefix: {file_prefix}")
        buffer = io.BytesIO()
        for filename in files:
            print(f"Loading chunk: {filename}")
            with open(filename, "rb") as f:
                buffer.write(f.read())
        buffer.seek(0) # Reset to start of buffer
        return torch.load(buffer, map_location=map_location, weights_only=False)
    state_dict = load_chunked("blobs/my_model_weights.pt", map_location="cpu")

    # load & check for a compiled state
    # remove the prefix if it was saved compiled
    uncompiled_state_dict = {}
    for key, value in state_dict["model_state_dict"].items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):] # Strip first 10 characters
            uncompiled_state_dict[new_key] = value
        else:
            uncompiled_state_dict[key] = value

    model.load_state_dict(uncompiled_state_dict)
    model = model.to(device_name)
    model.eval()
    return model

config, stoi, itos = get_resources()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_checkpoint_model(device)



### App Logic & Visualization ###

# helper
def get_html_token(text, color="black", tooltip=None, border=False):
    """Generates HTML span with proper escaping to prevent rendering bugs."""
    text = html.escape(text)
    # Replace spaces with non-breaking spaces so they are visible
    display_text = text.replace(" ", "&nbsp;") if text.strip() == "" else text
    
    style = f"color: {color}; padding: 2px 4px; border-radius: 4px; font-family: monospace; font-size: 1.2em;"
    if border:
        style += " border: 2px solid red; background-color: #ffe6e6;"
    else:
        style += " background-color: #f0f2f6;"
        
    title_attr = ""
    if tooltip:
        safe_tooltip = html.escape(tooltip).replace("\n", "&#10;")
        title_attr = f'title="{safe_tooltip}"'
    
    return f'<span style="{style}" {title_attr}>{display_text}</span>'

# Main Layout #
st.title("🧩 Visualizing Discrete Diffusion")
st.markdown("""
This tool visualizes **Masked Discrete Diffusion**. 
Instead of adding Gaussian noise (Standard Diffusion), we add "Mask Noise" (replacing tokens with `[MASK]`).
""")

# Sidebar #
st.sidebar.header("Configuration")
user_text = st.sidebar.text_input("Input Sentence", "To be, or not to be, that is the question:")
# Validate input length
if len(user_text) > config.block_size:
    st.sidebar.warning(f"Text truncated to {config.block_size} chars.")
    user_text = user_text[:config.block_size]
# Encode
encoded = [stoi.get(c, 0) for c in user_text]
seq_len = len(encoded)


st.header("1. The Math of Discrete Diffusion")

col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown("### A. The Forward Process (Implicit $Q_t$)")
    st.markdown(r"""
    We define a transition with an **Absorbing State**: a token either stays $x_0$ or becomes `[MASK]`. 
    This gives us the marginal distribution directly:
    """)
    st.latex(r"q(x_t | x_0) = \alpha_t \cdot \delta_{x_0} + (1 - \alpha_t) \cdot \delta_{\text{MASK}}")
    st.caption(r"At time $t$, probability $\alpha_t$ to see original token, $1-\alpha_t$ to see mask.")

    st.markdown("### B. The Reverse Process (Posterior)")
    st.markdown(r"""
    For any two times **$0 \le s < t \le 1$**, we model the transition $q(x_s | x_t, x_0)$.
    Since we don't know $x_0$, we use a Neural Network $\mu_\theta(x_t, t)$ to predict it.
    
    If $x_t$ is masked, the **Unmasking Probability** (rate of revealing) is:
    """)
    st.latex(r"p(unmask) = \frac{\alpha_s - \alpha_t}{1 - \alpha_t}")
    st.markdown(r"""
    We flip a coin with this probability. If successful, we replace `[MASK]` with $\mu_\theta$.
    """)

    st.markdown("### C. Training Objective (ELBO)")
    st.markdown(r"""
    The KL divergence collapses into a simple **Cross Entropy** term. 
    We pick a time $t$, mask the input, and optimize:
    """)
    st.latex(r"\mathcal{L} = \mathbb{E}_t \left[ w(t) \cdot \text{CE}\Big(x_0, \mu_\theta(x_t, t)\Big) \right]")
    st.markdown(r"Where the weighting term $w(t)$ emphasizes harder examples:")
    st.latex(r"w(t) = \frac{\alpha'(t)}{1 - \alpha(t)}")

# Interactive Controls & Schedule Plot
with col2:
    st.subheader("Visualization Control")
    
    with st.container(border=True):
        st.markdown(r"**Current Time $t$** (Controls masking level)")
        # t=1.0 is full noise, t=0.0 is clean data
        current_t = st.slider("Time (t)", 0.0, 1.0, 0.95, step=0.01, label_visibility="collapsed")

        st.markdown(r"**Discretization** (Number of Steps)")
        num_steps = st.slider("Steps", 4, 100, 12, label_visibility="collapsed")
    
    dt = 1.0 / num_steps

    # Generate 100 points for the curve
    t_vals = torch.linspace(0, 1, 100, device=device)
    # EXACT ALPHAS FROM THE MODEL ITESELF
    alpha_vals = model.get_alpha(t_vals).cpu().numpy()
    t_vals_np = t_vals.cpu().numpy()
    
    # Get scalar alpha for the current slider position
    current_alpha_tensor = model.get_alpha(torch.tensor([current_t], device=device))
    current_alpha = current_alpha_tensor.item()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_vals_np, y=alpha_vals, mode='lines', name='Alpha (Signal)', line=dict(width=3, color='#00CC96')))
    
    # Add the red dashed line for current time
    fig.add_shape(type="line", 
                  x0=current_t, y0=0, 
                  x1=current_t, y1=1, 
                  line=dict(color="#FF4B4B", width=2, dash="dash"))
    
    # Add a marker showing the current value
    fig.add_trace(go.Scatter(
        x=[current_t], y=[current_alpha],
        mode='markers', marker=dict(color='#FF4B4B', size=12),
        name='Current State',
        showlegend=False
    ))

    fig.update_layout(
        title="<b>Cosine Schedule</b> (Signal Strength)", 
        xaxis_title="Time t (Noise Level)", 
        yaxis_title="Signal α<sub>t</sub>", 
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        template="plotly_dark"
    )
    
    fig.add_annotation(
        x=0.5, y=0.5, 
        text="α<sub>t</sub> = 1 - cos(π/2(1-t))", 
        showarrow=False, 
        font=dict(size=14, color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Deterministic Masking Logic
# Use fixed seed so tokens don't flicker while dragging slider
rng = torch.Generator().manual_seed(42) 
# These thresholds represent the random variable 'u' ~ Uniform[0,1] for each token
mask_thresholds = torch.rand(seq_len, generator=rng)

# LOGIC CHANGE: 
# Alpha is the probability of keeping the token (Signal).
# A token is masked if random_val > alpha.
# t=0 -> alpha=1 -> u > 1 is False (All visible)
# t=1 -> alpha=0 -> u > 0 is True  (All masked)
is_masked = mask_thresholds > current_alpha

# Run Model Inference
input_ids = torch.tensor(encoded, device=device).unsqueeze(0)
masked_input = input_ids.clone()
masked_input[0, is_masked] = model.mask_token_id

with torch.no_grad():
    t_tensor = torch.tensor([current_t], device=device)
    # Model expects [B, T] input and [1] time
    logits = model(masked_input, t_tensor) 
    probs = F.softmax(logits, dim=-1)

# --- Tabs ---
tab_mask, tab_unmask, tab_train = st.tabs(["👁️ Forward Process", "✨ Reverse Process", "🎓 Training Objective"])

with tab_mask:
    st.subheader("Forward Process: Adding Noise")
    st.write(f"At time $t={current_t:.2f}$ (Signal $\\alpha={current_alpha:.2f}$), tokens are masked if their random threshold $u > \\alpha_t$.")
    html_tokens = []
    for i, char in enumerate(user_text):
        if is_masked[i]:
            html_tokens.append(get_html_token("[MASK]", color="lightgrey"))
        else:
            html_tokens.append(get_html_token(char, color="black"))
    st.markdown("".join(html_tokens), unsafe_allow_html=True)
    st.info("💡 Drag 'Time (t)' from 0.0 to 1.0. Notice how the Cosine schedule keeps tokens visible longer at the start compared to linear.")

with tab_unmask:
    st.subheader("Reverse Process: One Step of Generation")
    # Calculate next step properties
    next_t = max(0.0, current_t - dt)
    next_alpha_tensor = model.get_alpha(torch.tensor([next_t], device=device))
    next_alpha = next_alpha_tensor.item()
    st.write(f"Simulating step from $t={current_t:.2f}$ ($\\alpha={current_alpha:.2f}$) down to $s={next_t:.2f}$ ($\\alpha={next_alpha:.2f}$).")

    # A token is revealed if it is currently masked (u > current_alpha)
    # BUT it should be visible at the next step (u <= next_alpha)
    # Since we are going t -> 0, alpha is INCREASING.
    will_unmask = (mask_thresholds > current_alpha) & (mask_thresholds <= next_alpha)
    
    html_tokens = []
    for i, char in enumerate(user_text):
        # Prepare Tooltip with Top 5 predictions
        topk_probs, topk_indices = torch.topk(probs[0, i], 5)
        tooltip = "\n".join([f"{itos[idx.item()]}: {p.item():.1%}" for p, idx in zip(topk_probs, topk_indices)])
        if is_masked[i]:
            if will_unmask[i]:
                # Being revealed now
                html_tokens.append(get_html_token("[REVEAL]", color="#2e7d32", tooltip=f"Top predictions:\n{tooltip}", border=True))
            else:
                # Stays masked
                html_tokens.append(get_html_token("[MASK]", color="lightgrey", tooltip=f"Predictions (ignored):\n{tooltip}"))
        else:
            # Already visible
            html_tokens.append(get_html_token(char, color="black"))

    st.markdown("".join(html_tokens), unsafe_allow_html=True)
    st.success("Hover over **Red Boxed** tokens. These are being unmasked because the Signal $\\alpha$ increased enough to cover their random threshold.")

with tab_train:
    st.subheader("Training Calculation")
    st.markdown("In training, we calculate Cross Entropy Loss **only** on the tokens that are currently masked.")
    
    data = []
    total_loss = 0
    count = 0

    for i, char in enumerate(user_text):
        true_idx = stoi.get(char, 0)
        
        # Get probability assigned to the CORRECT character
        prob_of_truth = probs[0, i, true_idx].item()
        
        # Get what the model thinks is most likely
        pred_idx = torch.argmax(probs[0, i]).item()
        pred_char = itos[pred_idx]
        
        if is_masked[i]:
            token_loss = -math.log(prob_of_truth + 1e-9)
            total_loss += token_loss
            count += 1
            status = "👺 MASKED (Loss)"
        else:
            token_loss = 0.0
            status = "✅ VISIBLE (Skip)"
            
        data.append({
            "Char": char,
            "State": status,
            "Pred": pred_char,
            "P(Truth)": f"{prob_of_truth:.4f}",
            "Loss": f"{token_loss:.4f}"
        })
        
    df = pd.DataFrame(data)

    def highlight_masked(row):
        return ['background-color: #ffcccc' if "MASKED" in row['State'] else '' for _ in row]

    st.dataframe(df.style.apply(highlight_masked, axis=1), use_container_width=True)
    
    if count > 0:
        st.metric("Batch Loss (Average)", f"{total_loss / count:.4f}")
    else:
        st.write("No masked tokens at t=0.")

col1, col2 = st.sidebar.columns(2)
from gif import generate_gif_from_streamlit, generate_reverse_gif_from_streamlit
with col1:
    if st.button("🎬 Forward GIF"):
        with st.spinner("Creating forward GIF..."):
            gif_path = generate_gif_from_streamlit(user_text, config, stoi, model, device)
            st.success("✅ Forward GIF created!")
            st.image(gif_path)
            with open(gif_path, "rb") as f:
                st.download_button("⬇️ Download", f, "forward.gif", "image/gif")

with col2:
    if st.button("✨ Reverse GIF"):
        with st.spinner("Creating reverse GIF..."):
            gif_path = generate_reverse_gif_from_streamlit(
                user_text, config, stoi, itos, model, device
            )
            st.success("✅ Reverse GIF created!")
            st.image(gif_path)
            with open(gif_path, "rb") as f:
                st.download_button("⬇️ Download", f, "reverse.gif", "image/gif")

# Random Sampling Section
st.markdown("---")
st.header("🎲 Generate Random Samples")
st.markdown("Sample new sequences from the model by running the full reverse diffusion process.")

sample_col1, sample_col2 = st.columns([1, 2])

with sample_col1:
    sample_length = st.number_input("Sequence Length", min_value=1, max_value=128, value=128, step=1)
    sample_steps = st.number_input("Diffusion Steps", min_value=10, max_value=200, value=50, step=10)
    
    if st.button("🎲 Generate Sample", type="primary"):
        with st.spinner(f"Generating {sample_length} character sequence..."):
            progress_bar = st.progress(0)
            
            # Generate using model's method
            generated_tokens = model.generate(seq_len=sample_length, steps=sample_steps)
            
            progress_bar.progress(1.0)
            
            # Decode the result
            generated_text = "".join([itos[int(tok)] for tok in generated_tokens])
            
            # Store in session state
            if 'generated_samples' not in st.session_state:
                st.session_state.generated_samples = []
            st.session_state.generated_samples.append(generated_text)

with sample_col2:
    st.subheader("Generated Samples")
    
    if 'generated_samples' in st.session_state and len(st.session_state.generated_samples) > 0:
        for idx, sample in enumerate(reversed(st.session_state.generated_samples[-5:])):  # Show last 5
            with st.container(border=True):
                st.markdown(f"**Sample {len(st.session_state.generated_samples) - idx}**")
                st.code(sample, language=None)
        
        if st.button("🗑️ Clear All Samples"):
            st.session_state.generated_samples = []
            st.rerun()
    else:
        st.info("Click 'Generate Sample' to create new sequences from the model.")
