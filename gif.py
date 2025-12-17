import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np

"""
Contains the functions to reproduce the GIFs from the presentation.
"""


def create_forward_process_gif(
    text, 
    stoi, 
    model, 
    device,
    output_path="forward_process.gif",
    num_frames=50,
    duration=100,  # ms per frame
    seed=42
):
    """
    Creates a GIF showing the forward masking process from t=0 to t=1.
    
    Args:
        text: Input string to visualize
        stoi: Character to index mapping
        model: Your MD4 model (needs get_alpha method)
        device: torch device
        output_path: Where to save the GIF
        num_frames: Number of frames in the animation
        duration: Milliseconds per frame
        seed: Random seed for deterministic masking thresholds
    """
    
    # encode text
    encoded = [stoi.get(c, 0) for c in text]
    seq_len = len(encoded)
    
    # Generate random thresholds
    rng = torch.Generator().manual_seed(seed)
    mask_thresholds = torch.rand(seq_len, generator=rng)

    frames = []
    t_values = np.linspace(0, 1, num_frames)    
    for t in t_values:
        # alpha for this timestep
        current_alpha_tensor = model.get_alpha(torch.tensor([t], device=device))
        current_alpha = current_alpha_tensor.item()

        # which tokens are masked?
        is_masked = mask_thresholds > current_alpha
        
        # visualization
        frame = create_frame(text, is_masked, t, current_alpha)
        frames.append(frame)

    # save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f" GIF saved to {output_path}")
    return output_path


def create_reverse_process_gif(
    text,
    stoi,
    itos,
    model,
    device,
    output_path="reverse_process.gif",
    num_steps=50,
    duration=100,
    seed=42,
    show_predictions=True
):
    """
    Creates a GIF showing the reverse unmasking process from t=1 to t=0.
    The model progressively reveals tokens by predicting their values.
    
    Args:
        text: Original text (for reference/comparison)
        stoi: Character to index mapping
        itos: Index to character mapping
        model: Your MD4 model
        device: torch device
        output_path: Where to save the GIF
        num_steps: Number of denoising steps
        duration: Milliseconds per frame
        seed: Random seed for deterministic masking
        show_predictions: Whether to show model predictions for masked tokens
    """
    
    # encode text
    encoded = [stoi.get(c, 0) for c in text]
    seq_len = len(encoded)
    
    # generate random thresholds
    rng = torch.Generator().manual_seed(seed)
    mask_thresholds = torch.rand(seq_len, generator=rng)

    # fully masked at t=1
    input_ids = torch.tensor(encoded, device=device).unsqueeze(0)
    current_state = torch.full_like(input_ids, model.mask_token_id)

    frames = []
    t_values = np.linspace(1, 0, num_steps)    
    for step_idx, current_t in enumerate(t_values):
        # alpha for current and next timestep
        current_alpha_tensor = model.get_alpha(torch.tensor([current_t], device=device))
        current_alpha = current_alpha_tensor.item()
        
        if step_idx < len(t_values) - 1:
            next_t = t_values[step_idx + 1]
            next_alpha_tensor = model.get_alpha(torch.tensor([next_t], device=device))
            next_alpha = next_alpha_tensor.item()
        else:
            next_t = 0.0
            next_alpha = 1.0
    
        # model predictions
        with torch.no_grad():
            t_tensor = torch.tensor([current_t], device=device)
            dtype = model.head.weight.data.dtype
            logits = model(current_state, t_tensor.to(dtype))
            probs = F.softmax(logits, dim=-1)
            predicted_tokens = torch.argmax(logits, dim=-1)
        
        # which tokens should be revealed this step?
        is_currently_masked = mask_thresholds > current_alpha
        will_be_revealed = (mask_thresholds > current_alpha) & (mask_thresholds <= next_alpha)
        
        # reveal tokens that should be unmasked
        for i in range(seq_len):
            if will_be_revealed[i]:
                current_state[0, i] = predicted_tokens[0, i]
        
        # visualization
        frame = create_reverse_frame(
            text=text,
            current_state=current_state[0].cpu(),
            is_masked=is_currently_masked,
            will_reveal=will_be_revealed,
            probs=probs[0].cpu() if show_predictions else None,
            itos=itos,
            t=current_t,
            alpha=current_alpha,
            step=step_idx + 1,
            total_steps=num_steps
        )
        frames.append(frame)

    # save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f" Reverse GIF saved to {output_path}")
    return output_path


def create_reverse_frame(text, current_state, is_masked, will_reveal, probs, itos, t, alpha, step, total_steps):
    """
    Creates a single frame for the reverse process showing unmasking.
    """
    # image settings
    char_width = 50
    char_height = 80
    padding = 20
    header_height = 100
    img_width = max(900, len(text) * char_width + 2 * padding)
    img_height = char_height + header_height + 2 * padding
    
    # create image
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try monospace font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        tiny_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 28)
            small_font = ImageFont.truetype("arial.ttf", 18)
            tiny_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            tiny_font = ImageFont.load_default()

    # header with t and alpha values
    header_text = f"Step {step}/{total_steps}  |  t = {t:.3f}  |  α(t) = {alpha:.3f}"
    draw.text((padding, padding), header_text, fill='black', font=small_font)

    # progress bar
    bar_y = padding + 30
    bar_width = img_width - 2 * padding
    bar_height = 20
    
    # background bar
    draw.rectangle(
        [padding, bar_y, padding + bar_width, bar_y + bar_height],
        outline='gray',
        width=2
    )

    # progress fill (1 - t because we're going backwards)
    fill_width = int(bar_width * (1 - t))
    draw.rectangle(
        [padding, bar_y, padding + fill_width, bar_y + bar_height],
        fill='#00CC96'
    )
    
    # Legend
    legend_y = padding + 60
    draw.rectangle([padding, legend_y, padding + 15, legend_y + 15], fill='#FFE6E6', outline='red', width=2)
    draw.text((padding + 20, legend_y - 2), "Revealing Now", fill='black', font=tiny_font)
    
    draw.rectangle([padding + 150, legend_y, padding + 165, legend_y + 15], fill='#E0E0E0', outline='gray')
    draw.text((padding + 170, legend_y - 2), "Still Masked", fill='black', font=tiny_font)
    
    draw.rectangle([padding + 300, legend_y, padding + 315, legend_y + 15], fill='#E8F5E9', outline='green')
    draw.text((padding + 320, legend_y - 2), "Revealed", fill='black', font=tiny_font)
    
    # draw text with masking/unmasking states
    y_pos = header_height + padding
    x_pos = padding
    
    for i, original_char in enumerate(text):
        current_token_id = current_state[i].item()

        if is_masked[i]:
            if will_reveal[i]:
                # Being revealed RIGHT NOW
                display_char = itos[current_token_id]
                bg_color = '#FFE6E6'
                text_color = '#C62828'
                border_color = 'red'
                border_width = 3
            else:
                # Still masked
                display_char = '[M]'
                bg_color = '#E0E0E0'
                text_color = '#888888'
                border_color = 'gray'
                border_width = 1
        else:
            # Already revealed
            display_char = itos[current_token_id]
            bg_color = '#E8F5E9'
            text_color = '#2E7D32'
            border_color = 'lightgreen'
            border_width = 1

        # draw background box
        box_padding = 3
        char_bbox = draw.textbbox((x_pos, y_pos), display_char, font=font)
        box_width = char_bbox[2] - char_bbox[0] + 2 * box_padding

        draw.rectangle(
            [x_pos - box_padding, y_pos - box_padding, 
             x_pos + box_width, y_pos + char_height - box_padding],
            fill=bg_color,
            outline=border_color,
            width=border_width
        )
    
        draw.text((x_pos, y_pos), display_char, fill=text_color, font=font)
        
        # Show top prediction probability for masked tokens
        if is_masked[i] and probs is not None:
            top_prob = probs[i].max().item()
            prob_text = f"{top_prob:.0%}"
            draw.text((x_pos, y_pos + 45), prob_text, fill='#666666', font=tiny_font)
    
        x_pos += char_width
    
    return img


def generate_reverse_gif_from_streamlit(user_text, config, stoi, itos, model, device):
    # Validate and truncate if needed
    if len(user_text) > config.block_size:
        user_text = user_text[:config.block_size]
    
    output_path = "reverse_diffusion.gif"
    
    create_reverse_process_gif(
        text=user_text,
        stoi=stoi,
        itos=itos,
        model=model,
        device=device,
        output_path=output_path,
        num_steps=50,
        duration=100
    )
    
    return output_path


def create_frame(text, is_masked, t, alpha):
    """
    Creates a single frame showing the text with masking.
    """
    # image settings
    char_width = 40
    char_height = 60
    padding = 20
    header_height = 80
    img_width = max(800, len(text) * char_width + 2 * padding)
    img_height = char_height + header_height + 2 * padding
    
    # create image
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try monospace font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 32)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 32)
            small_font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    # header with t and alpha values
    header_text = f"t = {t:.3f}  |  α(t) = {alpha:.3f}  |  Mask Rate = {1-alpha:.1%}"
    draw.text((padding, padding), header_text, fill='black', font=small_font)
    
    # progress bar
    bar_y = padding + 35
    bar_width = img_width - 2 * padding
    bar_height = 20

    # background bar
    draw.rectangle(
        [padding, bar_y, padding + bar_width, bar_y + bar_height],
        outline='gray',
        width=2
    )
    
    # progress fill
    fill_width = int(bar_width * t)
    draw.rectangle(
        [padding, bar_y, padding + fill_width, bar_y + bar_height],
        fill='#FF4B4B'
    )
    
    # Draw text with masking
    y_pos = header_height + padding
    x_pos = padding
    
    for i, char in enumerate(text):
        # Prepare display character
        display_char = '[M]' if is_masked[i] else char
        if is_masked[i]:
            bg_color = '#E0E0E0'
            text_color = '#888888'
        else:
            bg_color = '#F0F2F6'
            text_color = 'black'

        # draw background box
        box_padding = 2
        char_bbox = draw.textbbox((x_pos, y_pos), display_char, font=font)
        box_width = char_bbox[2] - char_bbox[0] + 2 * box_padding
        
        draw.rectangle(
            [x_pos - box_padding, y_pos - box_padding, 
             x_pos + box_width, y_pos + char_height - box_padding],
            fill=bg_color,
            outline='lightgray'
        )
    
        draw.text((x_pos, y_pos), display_char, fill=text_color, font=font)
        
        x_pos += char_width
    
    return img


def generate_gif_from_streamlit(user_text, config, stoi, model, device):
    if len(user_text) > config.block_size:
        user_text = user_text[:config.block_size]
    
    output_path = "forward_diffusion.gif"
    
    create_forward_process_gif(
        text=user_text,
        stoi=stoi,
        model=model,
        device=device,
        output_path=output_path,
        num_frames=50,
        duration=100
    )
    
    return output_path