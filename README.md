# Crop Disease Detector 🌱
 
Deep learning model for plant disease classification using ResNet-50.
 
## 🎯 Model Performance
 
- **Architecture**: ResNet-50 (pretrained on ImageNet)
- **Best Validation Accuracy**: Check `logs/training_log.json`
- **Training Dataset**: PlantVillage (38 classes)
- **Fine-tuning Strategy**: Layers 3, 4, and classifier
 
## 📊 Dataset
 
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Classes**: 38 plant disease categories
- **Split**: 70% train, 20% validation, 10% test
- **Preprocessing**: ImageNet normalization
 
## 🏗️ Architecture Details
 
```
ResNet-50 (Pretrained on ImageNet)
├── Frozen: conv1, bn1, layer1, layer2
├── Fine-tuned: layer3, layer4
└── Classifier: Dropout(0.3) + Linear(2048 → 38)
```
 
## 📈 Training Configuration
 
- **Epochs**: 15
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: StepLR (step_size=7, gamma=0.1)
- **Batch Size**: 32
- **Data Augmentation**:
  - Random horizontal flip (50%)
  - Random vertical flip (30%)
  - Random rotation (±30°)
  - Color jitter
  - Random crop
 
## 📁 Project Structure
 
```
crop-disease-detector/
├── data/
│   ├── processed/
│   │   ├── train/    # Training images
│   │   ├── val/      # Validation images
│   │   └── test/     # Test images
│   └── class_names.json
├── models/
│   └── best_model.pth
├── logs/
│   ├── training_log.json
│   └── training_curves.png
└── README.md
```
 
## 🚀 Quick Start
 
### Training (Google Colab)
 
1. Clone repository:
```python
!git clone https://github.com/karthik-the-legend/crop-disease-detector.git
%cd crop-disease-detector
```
 
2. Download dataset and train (see notebook cells)
 
### Inference
 
```python
import torch
from torchvision import models, transforms
from PIL import Image
 
# Load model
checkpoint = torch.load('models/best_model.pth')
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, 38)
)
model.load_state_dict(checkpoint['model_state'])
model.eval()
 
# Preprocess image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
# Predict
image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(1).item()
    
print(f"Predicted: {checkpoint['class_names'][pred_class]}")
```
 
## 📊 Results
 
Training curves and detailed metrics available in `logs/training_log.json`
 
## 🛠️ Requirements
 
```
torch>=2.0.0
torchvision>=0.15.0
tqdm
matplotlib
PIL
```
 
## 📝 License
 
MIT License
 
## 👤 Author
 
**Karthik**
- GitHub: [@karthik-the-legend](https://github.com/karthik-the-legend)
 
---
*Trained on Google Colab with GPU acceleration*
