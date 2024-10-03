import torch
import requests
import cv2
import numpy as np

from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize video capture
cap = cv2.VideoCapture(0)

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)

# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load model weights
model.load_state_dict(torch.load('clipseg/weights/rd64-uni.pth', map_location=device), strict=False)


# Transform for the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])

# Define the prompts
prompts = ['human']
n = len(prompts)

# Initialize the plot
fig, ax = plt.subplots(1, n+1, figsize=(15, 4))
[axis.axis('off') for axis in ax.flatten()]


while 1:
    ret, frame = cap.read()
    if not ret:
        break
    
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(img.repeat(n, 1, 1, 1), prompts)[0]
    
    # Display original image
    cv2.imshow('Original', frame)
    
    # Display predictions
    for i in range(n):
        pred_mask = torch.sigmoid(preds[i][0]).cpu().numpy()
        pred_mask = (pred_mask * 255).astype(np.uint8)
        cv2.imshow(f'Prediction: {prompts[i]}', pred_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()


# Release the video capture when done
cap.release()