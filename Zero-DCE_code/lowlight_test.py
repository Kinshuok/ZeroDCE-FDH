import torch
import torch.nn as nn
import torchvision
import os
import argparse
import time
import numpy as np
from PIL import Image
import glob
from torchvision import transforms
import model  # Import your custom model module


def lowlight(image_path):
    """
    Enhance a low-light image using the trained model.

    Args:
        image_path (str): Path to the low-light input image.

    Saves:
        Enhanced image to the result folder corresponding to the input image path.
    """
    # Set the device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cpu")  # Change to "cuda" if GPU is available

    # Load the input image
    data_lowlight = Image.open(image_path)
    data_lowlight = np.asarray(data_lowlight) / 255.0  # Normalize image to [0, 1]
    data_lowlight = torch.from_numpy(data_lowlight).float()  # Convert to tensor
    data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0).to(device)  # Reshape to (1, C, H, W)

    # Load the trained model
    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth', map_location=device))  # Map model to device
    DCE_net.eval()  # Set the model to evaluation mode

    # Enhance the image
    start_time = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)  # Forward pass through the model
    end_time = time.time() - start_time
    print(f"Inference time: {end_time:.4f} seconds")

    # Save the enhanced image
    result_path = image_path.replace('test_data', 'result2')  # Update path for saving
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))
    torchvision.utils.save_image(enhanced_image, result_path)
    print(f"Saved enhanced image to: {result_path}")


if __name__ == '__main__':
    """
    Main script for enhancing all images in the test data directory.
    """
    # Disable gradient computation for inference
    with torch.no_grad():
        file_path = 'data/test_data/'  # Directory containing test images
        file_list = os.listdir(file_path)

        # Iterate through subdirectories and images
        for file_name in file_list:
            test_list = glob.glob(file_path + file_name + "/*")
            for image in test_list:
                print(f"Processing: {image}")
                lowlight(image)
