
![fxhgdh.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/5Rnksm-CjGsEs6XMZB-su.png)

#  **Fire-Detection-Siglip2**  

>  **Fire-Detection-Siglip2** is an image classification vision-language encoder model fine-tuned from google/siglip2-base-patch16-224 for a single-label classification task. It is designed to detect fire, smoke, or normal conditions using the SiglipForImageClassification architecture.  

    Classification report:
    
                  precision    recall  f1-score   support
    
            fire     0.9940    0.9881    0.9911      1010
          normal     0.9892    0.9941    0.9916      1010
           smoke     0.9990    1.0000    0.9995      1010
    
        accuracy                         0.9941      3030
       macro avg     0.9941    0.9941    0.9941      3030
    weighted avg     0.9941    0.9941    0.9941      3030


The model categorizes images into three classes:  
- **Class 0:** "Fire" – The image shows active fire.  
- **Class 1:** "Normal" – The image depicts a normal, fire-free environment.  
- **Class 2:** "Smoke" – The image contains visible smoke, indicating potential fire risk.  

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```


# **Intended Use:**  

The **Fire-Detection-Siglip2** model is designed to classify images into three categories: **fire, smoke, or normal conditions**. It helps in early fire detection and environmental monitoring.  

### Potential Use Cases:  
- **Fire Safety Monitoring:** Detecting fire and smoke in surveillance footage.  
- **Early Warning Systems:** Helping in real-time fire hazard detection in public and private areas.  
- **Disaster Prevention:** Assisting emergency response teams by identifying fire-prone areas.  
- **Smart Home & IoT Integration:** Enhancing automated fire alert systems in smart security setups.  
