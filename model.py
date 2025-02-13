import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

def get_model():
    """Load and return the pre-trained model"""

    # Load pre-trained model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    def predict(image):
        """
        Make prediction on an input image
        Args:
            image: numpy array in RGB format
        Returns:
            predictions: model predictions
        """
        try:
            # Convert numpy array to PIL Image if necessary
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image

            # Convert to tensor and normalize
            img_tensor = F.to_tensor(image_pil)

            # Add batch dimension and move to device
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                predictions = model(img_tensor)

            return predictions

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

    return predict