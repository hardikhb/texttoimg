import torch
from diffusers import StableDiffusionPipeline

# Load the pre-trained Stable Diffusion model
model_id = "stabilityai/stable-diffusion-2"  # You can change this to another model if you want
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)


# Move the model to GPU if available
if torch.cuda.is_available():
    pipe.to("cuda")
else:
    print("Warning: CUDA not available, using CPU. This may be slow.")

# Get user input for the text prompt
prompt = input("Enter your prompt for image generation: ")

# Fixed parameters for consistency
NUM_INFERENCE_STEPS = 50  # Higher steps = better quality but slower
GUIDANCE_SCALE = 14 # Controls how much the model follows the prompt

# Generate the image
image = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE).images[0]

# Save the generated image
image.save("generated_image.png")
image.show()

print("Image generated and saved as 'generated_image.png'")  