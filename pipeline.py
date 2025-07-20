import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import torchvision.models as models

class DigitRecognizer:
    def __init__(
        self,
        obb_model_path: str,
        orientation_model_path: str,
        digit_model_path: str,
        orientation_classes: list = ['bot', 'left', 'right', 'top'],
        device: str = 'cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        # Stage 1: OBB Detector
        self.obb_model = YOLO(obb_model_path)
        # Stage 2: Orientation Classifier
        self.orientation_classes = orientation_classes
        self.orientation_model = models.resnet18()
        self.orientation_model.fc = torch.nn.Linear(self.orientation_model.fc.in_features, len(orientation_classes))
        self.orientation_model.load_state_dict(torch.load(orientation_model_path, map_location=self.device))
        self.orientation_model.eval()
        self.orientation_model.to(self.device)
        self.orientation_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        # Stage 3: Digit Detector
        self.digit_model = YOLO(digit_model_path)

    def detect_obb(self, image: np.ndarray):
        results = self.obb_model(image)
        obbs = []
        if hasattr(results[0], 'obb') and results[0].obb is not None:
            for obb in results[0].obb.xyxyxyxy:
                points = obb.cpu().numpy().astype('int32')
                rect = cv2.minAreaRect(points.reshape((-1, 1, 2)))
                box = cv2.boxPoints(rect)
                box = box.astype('int32')
                width = int(rect[1][0])
                height = int(rect[1][1])
                if width == 0 or height == 0:
                    continue
                src_pts = box.astype("float32")
                dst_pts = np.array([[0, height - 1],
                                    [0, 0],
                                    [width - 1, 0],
                                    [width - 1, height - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                cropped_img = cv2.warpPerspective(image, M, (width, height))
                obbs.append(cropped_img)
        return obbs

    def resize_and_pad(self, image: np.ndarray, target_size=(224, 224)):
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_img = cv2.resize(image, (new_w, new_h))
        pad_w = target_size[0] - new_w
        pad_h = target_size[1] - new_h
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_img

    def predict_orientation(self, image: np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_tensor = self.orientation_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.orientation_model(img_tensor)
            _, pred = torch.max(outputs, 1)
            # print(f"Orientation prediction!!!!!!!!!!!!!!!: {pred.item()}")
            # print(f"Orientation prediction index: {pred.item()}, class: {self.orientation_classes[pred.item()]}")
        return self.orientation_classes[pred.item()]

    def rotate_image_to_top(self, image: np.ndarray, orientation: str):
        if orientation == "left":
            print("Rotating image to right orientation.")
            print(f"Before rotation: {image.shape}")
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            print(f"After rotation: {rotated.shape}")
            # img_pil = Image.fromarray(rotated)
            # img_pil.save("debug_rotated_left.jpg")
            # cv2.imwrite("/after_rotate_left.jpg", rotated)
            return rotated
        elif orientation == "right":
            print("Rotating image to left orientation.")
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation == "bot":
            print("Rotating image to top orientation.")
            return cv2.rotate(image, cv2.ROTATE_180)
        print("Orientation is top, no rotation needed.")
        return image

    def preprocess(self, image: np.ndarray):
        # Unsharp mask
        image = image.astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        sharpened = image + 1.0 * (image - blurred)
        sharpened = np.clip(sharpened, 0, 1)
        sharpened = (sharpened * 255).astype(np.uint8)
        # Bilateral filter
        filtered = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)
        return filtered

    def detect_digits(self, image: np.ndarray):
        results = self.digit_model(image)
        # Check if results contain boxes
        if not hasattr(results[0], 'boxes') or results[0].boxes is None or len(results[0].boxes) == 0:
            return ""
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        class_names = self.digit_model.names
        digit_predictions = []
        for i in range(len(classes)):
            class_id = int(classes[i])
            class_name = class_names[class_id]
            x_center = (boxes[i][0] + boxes[i][2]) / 2
            digit_predictions.append((class_name, x_center))
        sorted_digits = sorted(digit_predictions, key=lambda item: item[1])
        sequence = "".join([digit for digit, _ in sorted_digits])
        return sequence

    def process_image(self, image_bytes: bytes, image_name: str):
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # Stage 1: OBB detection and cropping
        obb_crops = self.detect_obb(img)
        if not obb_crops:
            print("No OBB detected.")
            return ""
        # For each detected region, process through orientation and digit detection
        sequences = []
        for crop in obb_crops:
            # Stage 2: Orientation correction (rotate first)
            orientation = self.predict_orientation(crop)
            print(f"Predicted orientation: {orientation}")
            cv2.imwrite(f"debug/before-rotate-images/{image_name}", crop)
            print(f"Crop shape before rotation: {crop.shape}")
            rotated = self.rotate_image_to_top(crop, orientation)
            print(f"Crop shape after rotation: {rotated.shape}")
            cv2.imwrite(f"debug/after-rotate-images/{image_name}", rotated)
            # Now resize and pad after rotation
            rotated_resized = self.resize_and_pad(rotated, target_size=(224, 224))
            # Preprocess
            processed = self.preprocess(rotated_resized)
            cv2.imwrite(f"debug/preprocess-images/{image_name}", processed)
            # Stage 3: Digit detection
            seq = self.detect_digits(processed)
            print(f"Digits detected: {seq}")
            if seq:
                sequences.append(seq)
        # If multiple crops, join with comma or return the first
        return ",".join(sequences) if sequences else ""