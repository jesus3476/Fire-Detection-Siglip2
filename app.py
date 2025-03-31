import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Fire-Detection-Siglip2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def fire_detection(image):
    """Classifies an image as fire, smoke, or normal conditions."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = model.config.id2label
    predictions = {labels[i]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=fire_detection,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Detection Result"),
    title="Fire Detection Model",
    description="Upload an image to determine if it contains fire, smoke, or a normal condition."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
