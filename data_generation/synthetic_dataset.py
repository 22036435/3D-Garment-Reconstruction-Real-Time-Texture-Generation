import openai
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import json
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Using the model-agnostic default `max_length`")

openai.api_key = "paste your open-AI API key here"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_basic_descriptions_from_images(image_paths):
    basic_descriptions = []
    for image_path in image_paths:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        basic_descriptions.append(caption)
    return basic_descriptions

def refine_descriptions_with_gpt(descriptions):
    refined_prompts = []
    for description in descriptions:
        retry_count = 0
        while retry_count < 5:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a creative assistant generating detailed and conceptual fashion-related prompts, focusing on garments, fabrics, and texture features."},
                        {"role": "user", "content": f"Refine this basic description into a detailed and creative fashion prompt: {description}"}
                    ],
                    max_tokens=200,
                    temperature=0.8,
                    n=1,
                )
                prompt = response['choices'][0]['message']['content'].strip()
                refined_prompts.append(prompt)
                break
            except openai.error.RateLimitError:
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    return refined_prompts

# Directory where images are stored
image_directory = r"C:\Users\22036435\Desktop\data\prompt_feed"

# Load reference images and generate refined descriptive prompts
def load_images_and_generate_prompts(image_directory):
    if os.path.exists(image_directory):
        image_paths = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if img.endswith(('.png', '.jpg', '.jpeg'))]
        basic_descriptions = generate_basic_descriptions_from_images(image_paths)
        return refine_descriptions_with_gpt(basic_descriptions)
    else:
        print(f"Image directory {image_directory} does not exist.")
        return []

# Generate descriptive prompts from the reference images
reference_image_prompts = load_images_and_generate_prompts(image_directory)

# Example initial prompts focused on fashion garments, fabrics, and texture features
initial_prompts = [
    "Imagine a luxurious evening gown made from a silky smooth fabric with a subtle sheen that catches the light. The material should be lightweight yet durable, perfect for creating a flowing silhouette. Consider incorporating intricate patterns inspired by nature, such as delicate floral motifs or abstract waves, to add a touch of sophistication and timeless beauty.",
    "Envision a bold, statement-making jacket designed for high fashion. The fabric should have a rich, tactile texture, reminiscent of hand-woven artistry. The material should be thick and plush, with a matte finish that exudes understated elegance. Think of patterns that merge traditional craftsmanship with contemporary design, such as geometric shapes intertwined with organic forms, to create a garment that is both visually striking and deeply symbolic.",
    "Conceptualize a fashion-forward skirt that marries the softness of velvet with the iridescence of silk. The texture should be velvety to the touch, with a luminous quality that changes hue under different lighting. The material should be versatile enough to be used in both structured and flowing designs. Consider patterns that evoke the mystery of the cosmos, with swirling galaxies and stardust forming an ethereal, otherworldly design.",
]

# Combine the reference image prompts with the initial prompts
combined_prompts = initial_prompts + reference_image_prompts

# Function to generate many diverse prompts based on combined initial examples
def generate_prompts(initial_prompt, num_variations=10):
    generated_prompts = []
    for _ in range(num_variations):
        retry_count = 0
        while retry_count < 5:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a creative assistant generating fashion-related prompts focused on garments, fabrics, and texture features."},
                        {"role": "user", "content": f"Generate a fashion prompt focused on garments, fabrics, and texture features, similar in style to the following: {initial_prompt}"}
                    ],
                    max_tokens=200,
                    temperature=0.8,
                    n=1,
                )
                prompt = response['choices'][0]['message']['content'].strip()
                generated_prompts.append(prompt)
                break
            except openai.error.RateLimitError:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    return generated_prompts

def generate_many_prompts(combined_prompts, total_prompts=500):
    all_generated_prompts = []
    num_variations_per_prompt = total_prompts // len(combined_prompts)

    for prompt in combined_prompts:
        prompts = generate_prompts(prompt, num_variations=num_variations_per_prompt)
        all_generated_prompts.extend(prompts)
        save_prompts_to_json(all_generated_prompts, "generated_fashion_prompts.json")
    
    return all_generated_prompts

def save_prompts_to_json(prompts, filename):
    with open(filename, "w") as f:
        json.dump(prompts, f, indent=4)

generated_prompts = generate_many_prompts(combined_prompts, total_prompts=500)

print(f"Final generated prompts have been saved to generated_fashion_prompts.json")
