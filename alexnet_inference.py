# # Import necessary libraries
import argparse
import numpy as np
from torchvision import transforms
import torch

from PIL import Image
# Load the pre-trained AlexNet model
from torchvision.models import alexnet
import json
import matplotlib.pyplot as plt

def load_image(image_path):
    image = Image.open(image_path)
    return image

def load_class_labels(file_path='imagenet_classes.json'):
    with open(file_path, 'r') as f:
        class_labels = json.load(f)
    return class_labels

def preprocess_image(image_path):
    # Define the image preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def load_model(weights='IMAGENET1K_V1'):
    # Load the pre-trained AlexNet model
    model = alexnet(weights=weights)
    # Set the model to evaluation mode
    model.eval()
    return model


def predict(image: torch.Tensor, model: torch.nn.Module):
    # Pass the image through the model
    with torch.no_grad():
        output = model(image)
    return output

def get_top_predictions(outputs,class_labels, top_k=3):
    # Get probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    # Get top predictions
    top_probabilities, top_indices = torch.topk(probabilities, top_k)
    
    predictions = []
    for i in range(top_k):
        pred_class = top_indices[0][i].item()
        pred_prob = top_probabilities[0][i].item()
        predictions.append(
            {'class': class_labels[str(pred_class)], 'probability': pred_prob}
        )

    return predictions

# visualizetion of probabilitiy using bar char
def visualize_predictions(top_predictions,save_path='top_predictions.png'):
    labels = [item['class'] for item in top_predictions]
    probabilities = [item['probability'] for item in top_predictions]


    # Create a bar chart
    plt.bar(labels, probabilities)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Top Predictions')

    # Save the chart
    plt.savefig(save_path)

    # Close the chart
    plt.close()
  



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,help='path to image')
    parser.add_argument('--weights', type=str, default='IMAGENET1K_V1', help='path to model weights')
    args = parser.parse_args()
    image_path = args.image
    weights = args.weights
    # Load the pre-trained AlexNet model
    model = load_model(weights)
    # Load the image
    image = load_image(image_path)
    # Preprocess the image
    image = preprocess_image(image_path)
    # Pass the image through the model
    outputs = predict(image, model)
    # Get the top predictions
    top_predictions = get_top_predictions(outputs, load_class_labels())
    # Visualize the top predictions
    visualize_predictions(top_predictions)

    


    



   

    