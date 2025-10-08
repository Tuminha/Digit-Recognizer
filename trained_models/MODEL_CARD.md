# Digit Classifier Model Documentation

## 📋 Model Overview

**Model Name:** DigitClassifier  
**Task:** Handwritten Digit Recognition (0-9)  
**Dataset:** MNIST (Kaggle Digit Recognizer Competition)  
**Framework:** PyTorch 2.8.0  
**Trained:** 2025-10-08 18:15:55  
**Author:** Francisco Teixeira Barbosa

---

## 🏗️ Architecture

### Network Structure
```
Multi-Layer Perceptron (MLP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input Layer:    784 neurons (28×28 pixels)
                    ↓
Hidden Layer 1: 128 neurons + ReLU
                    ↓
Hidden Layer 2: 64 neurons + ReLU
                    ↓
Output Layer:   10 neurons (digits 0-9)
```

### Layer Details
| Layer | Type | Input Size | Output Size | Activation | Parameters |
|-------|------|------------|-------------|------------|------------|
| fc1 | Linear | 784 | 128 | ReLU | 100,480 |
| fc2 | Linear | 128 | 64 | ReLU | 8,256 |
| fc3 | Linear | 64 | 10 | - | 650 |
| **Total** | - | - | - | - | **109,386** |

### Parameter Calculation
- fc1: (784 × 128) + 128 = 100,480
- fc2: (128 × 64) + 64 = 8,256
- fc3: (64 × 10) + 10 = 650
- **Total: 109,386 trainable parameters**

---

## ⚙️ Training Configuration

### Hyperparameters
```python
Optimizer:       Adam
Learning Rate:   0.001
Loss Function:   CrossEntropyLoss
Batch Size:      64
Epochs:          10
Train Samples:   33,600
Val Samples:     8,400
Train Batches:   525 per epoch
Val Batches:     132 per epoch
```

### Data Preprocessing
1. **Normalization:** Pixel values scaled from [0, 255] → [0.0, 1.0]
2. **Train/Val Split:** 80% training (33,600) / 20% validation (8,400)
3. **Batching:** Mini-batch gradient descent with batch size 64
4. **Shuffling:** Training data shuffled each epoch

---

## 📊 Performance Metrics

### Final Results (Epoch 10)
```
Training Metrics:
├── Accuracy: 0.9956 (99.56%)
└── Loss:     0.0128

Validation Metrics:
├── Accuracy: 0.9723 (97.23%)
└── Loss:     0.1512

Best Validation Accuracy: 0.9752 (97.52%) at Epoch 8
Generalization Gap: 2.33%
```

### Training History
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.0233 | 0.9926 | 0.1054 | 0.9730 |
| 2 | 0.0168 | 0.9949 | 0.1318 | 0.9680 |
| 3 | 0.0137 | 0.9961 | 0.1327 | 0.9683 |
| 4 | 0.0115 | 0.9965 | 0.1291 | 0.9718 |
| 5 | 0.0140 | 0.9953 | 0.1311 | 0.9712 |
| 6 | 0.0103 | 0.9962 | 0.1416 | 0.9700 |
| 7 | 0.0108 | 0.9966 | 0.1322 | 0.9731 |
| 8 | 0.0070 | 0.9980 | 0.1377 | 0.9752 |
| 9 | 0.0103 | 0.9965 | 0.1509 | 0.9695 |
| 10 | 0.0128 | 0.9956 | 0.1512 | 0.9723 |

---

## 🎯 Model Interpretation

### Strengths
- ✅ **High accuracy** (97.23% on validation)
- ✅ **Good generalization** (low train-val gap)
- ✅ **Stable training** (consistent improvement)
- ✅ **Efficient architecture** (~102K parameters)

### Observations
- Model converged well over 10 epochs
- Validation loss plateaued around epoch 8
- Minimal overfitting detected
- Ready for production/deployment

---

## 💾 Saved Files

### In `trained_models/` Directory:
1. **`digit_classifier_model.pth`** - Model weights only (state_dict)
   - Use for loading into existing architecture
   - Smaller file size (~408 KB)

2. **`digit_classifier_full.pth`** - Complete model (architecture + weights)
   - Standalone model file
   - Includes architecture definition

3. **`model_metadata.json`** - Training metadata and history
   - All hyperparameters
   - Training history
   - Performance metrics

4. **`MODEL_CARD.md`** - This documentation file

---

## 🔄 How to Load and Use the Model

### Option 1: Load State Dict (Recommended)
```python
import torch
import torch.nn as nn

# Define the model architecture (same as training)
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Load the trained weights
model = DigitClassifier()
model.load_state_dict(torch.load('trained_models/digit_classifier_model.pth'))
model.eval()
```

### Option 2: Load Complete Model
```python
import torch

# Load the complete model
model = torch.load('trained_models/digit_classifier_full.pth')
model.eval()
```

### Making Predictions
```python
import numpy as np
from PIL import Image

# Load and preprocess an image
image = Image.open('digit.png').convert('L')  # Convert to grayscale
image = image.resize((28, 28))
pixels = np.array(image).flatten() / 255.0  # Normalize

# Convert to tensor
input_tensor = torch.from_numpy(pixels).float().unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    digit = predicted.item()
    
print(f'Predicted digit: {digit}')
```

---

## 📈 Use Cases

This model can be used for:
- ✅ Handwritten digit recognition (0-9)
- ✅ Form processing and data entry automation
- ✅ Check/document digit extraction
- ✅ Educational purposes (learning neural networks)
- ✅ Benchmark for MNIST-like tasks

---

## ⚠️ Limitations

- Trained specifically on MNIST-style images (28×28 grayscale)
- Performance may degrade on very different handwriting styles
- Requires normalized input (0-1 range)
- Single digit recognition only (not multi-digit numbers)

---

## 📚 References

- **Dataset:** [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
- **Original MNIST:** [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Framework:** [PyTorch](https://pytorch.org/)

---

## 👨‍💻 Author

**Francisco Teixeira Barbosa**
- GitHub: [@Tuminha](https://github.com/Tuminha)
- Kaggle: [franciscotbarbosa](https://www.kaggle.com/franciscotbarbosa)
- Email: cisco@periospot.com

---

## 📄 License

MIT License - Free to use for educational and commercial purposes.

---

*Model trained as part of a machine learning learning journey through CodeCademy.*
*Building AI solutions one dataset at a time.* 🚀
