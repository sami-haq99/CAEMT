from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
import gc

torch.set_float32_matmul_precision('high')

model_id = "./gemma-3-12b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id, use_fat = True)

# Directory containing your images
file_names = []
with open("../test-set/commute/en-de/img.order", "r") as file:
    file_names = [line.strip() for line in file if line.strip()]
    
file_names = file_names[92:]  # Limit to 1000 images for testing

output_file = "captions-commute-gemma3b92ow.de"

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

image_paths = ["../test-set/commute/images/images/" + file_name for file_name in file_names]

with open(output_file, "w", encoding="utf-8") as f:
    for filename in image_paths:
        if filename.lower().endswith(image_extensions):
            try:

                image = Image.open(filename).convert("RGB")
                max_size = (512, 512)

                # Only downscale if needed
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size)
                    
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Beschreiben Sie das Bild in nur einem Satz in deutscher Sprache."}
                        ]
                    }
                ]

                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                    generation = generation[0][input_len:]

                decoded = processor.decode(generation, skip_special_tokens=True)

                f.write(f"{decoded.strip()}\n")
                
                del inputs, generation
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")