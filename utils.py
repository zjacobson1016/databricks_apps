import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

def create_synthetic_image():
    """Create a synthetic image with car annotation"""
    # Create a white background
    img = Image.new('RGB', (600, 400), 'white')
    draw = ImageDraw.Draw(img)

    # Draw a simple car shape (rectangle)
    car_color = (100, 100, 100)
    draw.rectangle((200, 150, 400, 250), fill=car_color)

    # Draw wheels
    draw.ellipse((230, 230, 270, 270), fill='black')
    draw.ellipse((330, 230, 370, 270), fill='black')

    # Draw bounding box
    draw.rectangle((190, 140, 410, 260), outline=(0, 255, 0), width=2)

    # Add label
    draw.text((190, 110), "Car", fill=(0, 255, 0))

    return np.array(img)

def draw_predictions(image, predictions):
    """Draw bounding boxes and labels on the image"""
    # Convert numpy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image

    draw = ImageDraw.Draw(img)

    try:
        # Move predictions to CPU if they're on GPU
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Draw each prediction with confidence > 0.5
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5 and label == 3:  # 3 is typically the class ID for cars in COCO
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)

                # Add label and confidence score
                label_text = f"Car: {score:.2f}"
                draw.text((x1, y1-20), label_text, fill=(0, 255, 0))

    except Exception as e:
        print(f"Error drawing predictions: {str(e)}")
        raise

    return np.array(img)