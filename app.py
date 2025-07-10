from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

# Initialize the pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    # Safety check for empty prompts
    if not prompt.strip():
        raise gr.Error("Please enter a valid prompt")

    # Generate image
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
    return image

# Create interface
iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Describe the image you want to generate..."),
    outputs="image",
    title="AI Text-to-Image Generator",
    description="Enter a text prompt to generate an image using Stable Diffusion"
)

# Launch with explicit settings for Colab
iface.launch(
    share=True,  # Required for Colab
    debug=True,  # To see errors in Colab
    server_port=7860  # Specific port for Colab
)
