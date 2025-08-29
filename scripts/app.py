import gradio as gr
import torch
from gradio.themes.base import Base
from torchvision.datasets import Food101
from models import EffNetV2_S
from prepare_data import get_model_components
from class_names import FOOD101_CLASSES

# --- 1. Configuration ---
MODEL_PATH = "checkpoints/best-model-epoch=22-val_acc=0.8541.ckpt" 
MODEL_NAME = "EfficientNet_V2_S"

theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="blue",
).set(

    body_background_fill="#f2f2f2"
)

# --- 2. Load Model and Assets ---
print("Loading model and assets...")
model = EffNetV2_S.load_from_checkpoint(MODEL_PATH)
model.eval()

components = get_model_components(MODEL_NAME)
transforms = components["val_transforms"]
class_names = FOOD101_CLASSES 

print("Model and assets loaded successfully.")

# --- 3. Prediction Function ---
def predict(image):
    """
    Takes a PIL image, preprocesses it, and returns the model's top 3 predictions.
    """
    # 1. Preprocess the image and add a batch dimension
    input_tensor = transforms(image).unsqueeze(0)
    
    # 2. Move the input tensor to the same device as the model
    # This ensures both the model and the data are on the GPU.
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 3. Make a prediction
    with torch.no_grad():
        output = model(input_tensor)
        
    # 4. Post-process the output
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return confidences
    

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a Food Image"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    theme=theme,
    
    # UI Enhancements
    title="🍔 Food-101 Image Classifier 🍟",
    description=(
        "What's on your plate? Upload an image or try one of the examples below to classify it. "
        "This demo uses an EfficientNetV2-S model fine-tuned on the Food-101 dataset."
    ),
    article=(
        "<p style='text-align: center;'>A project by Daniel Kiani. "
        "<a href='https://github.com/Deathshot78/Food101-Classification' target='_blank'>Check out the code on GitHub!</a></p>"
    ),
    examples=[
        ["assets/ramen.jpg"],
        ["assets/pizza.jpg"],
        ["assets/oysters.jpg"],
        ["assets/onion_rings.jpg"]
    ]
)

# --- 5. Launch the App ---
if __name__ == "__main__":
    demo.launch(debug=True)