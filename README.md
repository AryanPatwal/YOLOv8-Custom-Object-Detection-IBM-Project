# YOLOv8 Custom Object Detection – IBM Project

## 📌 Project Description
This project demonstrates end-to-end training, validation, and inference of a custom object detection model using YOLOv8. It includes dataset setup, YAML configuration, model training, performance evaluation, and prediction generation. Built as part of an IBM project, it provides a clear workflow for real-world computer vision tasks.

---

## 🚀 Features
- Custom dataset preparation  
- YOLOv8 environment setup  
- Auto-generation of `data.yaml`  
- Model training with adjustable hyperparameters  
- Validation metrics and evaluation  
- Inference on test images  
- Organized notebook for learning and reproducibility  

---

## 📁 Repository Structure
```
├── IBM_project_1.ipynb      # Main notebook
├── yolomodel.zip            # Dataset/model archive (optional)
├── README.md                # Project documentation
└── results/                 # Training output, logs, predictions
```

---

## 🧩 Technologies Used
- Python 3  
- Ultralytics YOLOv8  
- PyTorch  
- NumPy  
- OpenCV  
- Matplotlib  
- Google Colab  

---

## 📦 Dataset Structure
```
dataset/
│
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## 🛠️ Training the Model
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/content/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.01,
    optimizer="SGD",
    val=True
)
```

---

## 🔍 Running Inference
```python
model = YOLO("runs/detect/train/weights/best.pt")
model.predict(source="/content/test.jpg", save=True)
```

Predictions are saved inside:
```
runs/detect/predict/
```

---

## 👤 Author
**Aryan Patwal**  
GitHub: *your link here*  
LinkedIn: *your link here*

---

## 📄 License
MIT License

---

## ⭐ Contribution
Contributions and suggestions are welcome.  
If this project helped you, please **star ⭐ the repository**!
