# Water Meter Digit Recognition

This project provides a pipeline for automatic water meter digit recognition using deep learning. It includes training scripts, model management, and an API for inference.

---

## 1. How to Train the Models
> **Training:**  We use **A100** in **Google colab** to training models in pipeline. So the directories below will be in your **Google colab notebook**. \
> **Annotate**: Using roboflow to annotate

All training steps are provided in the `biwase.ipynb` notebook. The pipeline consists of three stages:

### Stage 1: Oriented Bounding Box (OBB) Detection
- Train a YOLOv8 OBB model to detect the water meter region.
- Download the dataset and run the training cell in the notebook:
  ```python
  model = YOLO('yolov8m-obb.pt')
  results = model.train(data="path/to/data.yaml", epochs=10, batch=64, imgsz=640, single_cls=True)
  ```
- The best model weights will be saved in the `runs/obb/train/weights/` directory.

### Stage 2: Orientation Classification
- Train a ResNet18 classifier to predict the orientation of cropped meter images.
- Prepare the orientation dataset as described in the notebook.
- Run the training cell for the classifier. The best weights will be saved as `stage-2.pth`.

### Stage 3: Digit Detection
- Train a YOLOv8 model (OBB or standard) to detect digits within the cropped and oriented meter images.
- Follow the notebook to train and save the best weights in `runs/detect/train/weights/` or `runs/obb/train/weights/`.

---

## 2. How to Get Model Weights for the API

After training, copy the best model weights to the `model/` directory in your project:

- **Stage 1 (OBB Detector):**
  - Copy `runs/obb/train/weights/best.pt` to `model/Stage1-BoundingBox/model-1.pt`
- **Stage 2 (Orientation Classifier):**
  - Copy `stage-2.pth` to `model/Stage2-Orientation/model-2.pth`
- **Stage 3 (Digit Detector):**
  - Copy the best digit detection model (e.g., `runs/detect/train/weights/best.pt` or `runs/obb/train/weights/best.pt`) to `model/Stage3-Digit/model-3.pt`

> **Important:**  
> Make sure the class order for orientation matches between training and API (`orientation_classes` parameter).
> **The default setting is: ['bot', 'left', 'right', 'top']**
---

## 3. How to Run the API

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```
   or (if using the provided command):
   ```bash
   fastapi dev main.py
   ```

3. **API Usage:**
   - Send a POST request to `/predict` with one or more images as files.
     ```bash
     Recommend using fastapi docs: http://127.0.0.1:8000/docs 
     Or using postman desktop. Postman agent can be fail because it can not send your right local data format.
     ```

4. **Check the results:**
   - The API will return the predicted digit sequence for each image.

---

**Note:**  
- Ensure your model weights are in the correct paths as described above.
- For best results, use the same preprocessing and class order as in