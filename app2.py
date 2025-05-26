import os
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import torch.nn.functional as F  # Import for softmax
import matplotlib.pyplot as plt
def main():
    model_path = r"D:\alt+hackj\df\model"

    model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(model_path)

    image_path = r"C:\Users\nalin\OneDrive\Pictures\Camera Roll\WIN_20250218_23_05_21_Pro.jpg"
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    print("Input Image Shape:", inputs['pixel_values'].shape)  # Should be (1, 3, 224, 224) or similar

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()

    # Display probabilities
    for idx, prob in enumerate(probs):
        print(f"{model.config.id2label[idx]}: {prob:.4f}")

    # Confidence-based prediction
    threshold = 0.4  # Lowering the threshold for better acceptance
    predicted_class = max(range(len(probs)), key=lambda i: probs[i])

    if probs[predicted_class] < threshold:
        print("Prediction: Uncertain (below confidence threshold)")
    else:
        print(f"Prediction: {model.config.id2label[predicted_class]}")

        

    plt.bar(model.config.id2label.values(), probs)
    plt.title("Confidence Distribution")
    plt.ylabel("Probability")
    plt.show()


if __name__ == "__main__":
    main()
