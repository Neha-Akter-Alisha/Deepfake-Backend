import torch
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageChops, ImageEnhance
import os

class HybridDeepFakeDetector:
    def __init__(self):
        """Initializes the AI model for image deepfake detection."""
        # This model is ONLY for the image/face analysis
        self.image_model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes=1)
        self.image_model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_image_deepfake(self, image_path):
        """
        Uses the AI model to predict the probability of a standard image (like a face) being a deepfake.
        """
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = torch.sigmoid(self.image_model(img_tensor))
            probability = output.item()
        
        return probability

    def analyze_document_forgery(self, image_path, output_dir='static/processed'):
        """
        Uses Error Level Analysis (ELA) to detect potential manipulations in documents.
        This method does NOT use the AI model.
        """
        TEMP_ELA_FILE = os.path.join(output_dir, 'temp_ela.jpg')
        ELA_HEATMAP_FILE = os.path.join(output_dir, os.path.basename(image_path))
        
        original = Image.open(image_path).convert('RGB')

        # 1. Re-save the image at a known quality (e.g., 90%)
        # This creates a baseline for compression
        original.save(TEMP_ELA_FILE, 'JPEG', quality=90)
        temporary = Image.open(TEMP_ELA_FILE)

        # 2. Find the difference between the original and the re-saved version
        # Manipulated areas will have a much larger difference
        diff = ImageChops.difference(original, temporary)

        # 3. Enhance the difference to make it visible
        # This creates the ELA heatmap
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1 # Avoid division by zero
        
        scale = 255.0 / max_diff
        brightened = ImageEnhance.Brightness(diff).enhance(scale)

        # Save the visible ELA heatmap
        brightened.save(ELA_HEATMAP_FILE)

        # 4. Calculate a quantitative "Forgery Score"
        # We do this by measuring the average brightness of the difference image
        diff_array = np.array(diff)
        forgery_score = diff_array.mean() * 3 # Multiply by a factor to make the score more sensitive

        # Clean up the temporary file
        os.remove(TEMP_ELA_FILE)

        return forgery_score, os.path.basename(image_path)

    def simulate_heatmap_for_image(self, image_path, output_dir='static/processed'):
        """
        Creates a *simulated* heatmap for the image deepfake result for demonstration.
        This does not represent the model's actual focus.
        """
        image = cv2.imread(image_path)
        heatmap = np.zeros_like(image, dtype=np.uint8)

        h, w, _ = image.shape
        fake_region = np.random.randint(50, 150, (h, w), dtype=np.uint8)
        heatmap[:, :, 2] = fake_region # Add red tint

        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        processed_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(processed_path, overlay)
        
        return os.path.basename(image_path)

