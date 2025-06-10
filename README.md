# Brain Tumor Classification using VGG16 Transfer Learning

A deep learning project that classifies brain tumor types from MRI images using transfer learning with the pre-trained VGG16 architecture.

## 🧠 Project Overview

This project implements a convolutional neural network to classify brain MRI images into four distinct categories:
- **Glioma** - Tumors that occur in the brain and spinal cord
- **Meningioma** - Tumors that arise from the meninges (protective layers around the brain)
- **No Tumor** - Normal brain scans without any tumor presence
- **Pituitary** - Tumors that develop in the pituitary gland

## 🏗️ Model Architecture

The model leverages **VGG16** with transfer learning approach:
- **Base Model**: Pre-trained VGG16 (ImageNet weights) with frozen layers
- **Custom Layers**:
  - Flatten layer to convert 2D feature maps to 1D
  - Dense layer with 256 neurons and ReLU activation
  - Output layer with 4 neurons and softmax activation for classification

```python
VGG16 Base (Frozen) → Flatten → Dense(256, ReLU) → Dense(4, Softmax)
```

## 📁 Dataset Structure

```
brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 🛠️ Requirements

### Python Libraries
```bash
tensorflow>=2.0
keras
numpy
matplotlib
pillow
scikit-learn
```

### Installation
```bash
pip install tensorflow numpy matplotlib pillow scikit-learn
```

## 🚀 Getting Started

### 1. Data Preparation
The project includes a custom data preparation function that handles:
- Image rescaling (normalization to [0,1])
- Data augmentation with brightness variation
- Batch processing
- Categorical encoding for multi-class classification

```python
def prepare_the_datasets(train_datasets, validation_datasets, batch_size, image_size):
    # Training data generator with augmentation
    train_datasets_generator = ImageDataGenerator(
        rescale=1.0/255,
        brightness_range=(0.8, 1.2)  # Random brightness adjustment
    )
    
    # Validation data generator (only rescaling)
    validation_datasets_generator = ImageDataGenerator(rescale=1.0/255)
```

### 2. Model Configuration

**Key Parameters:**
- **Input Shape**: 224 × 224 × 3 (RGB images)
- **Batch Size**: Configurable (typically 32 or 64)
- **Image Size**: 224 pixels (standard for VGG16)
- **Classes**: 4 (categorical classification)

### 3. Training Process

```python
# Model compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    train_data, 
    validation_data=validation_data,
    epochs=10,
    steps_per_epoch=steps
)
```

## 📊 Model Performance

### Training Configuration
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Transfer Learning**: Frozen VGG16 base + custom classifier

### Visualization
The project includes performance visualization:
- Training vs Validation Accuracy plots
- Learning curve analysis
- Model performance tracking over epochs

## 🔍 Key Features

### Data Augmentation
- **Rescaling**: Normalizes pixel values to [0,1] range
- **Brightness Adjustment**: Random brightness variation (0.8-1.2 factor)
- **Shuffle**: Randomizes training data order

### Transfer Learning Benefits
- **Pre-trained Features**: Leverages ImageNet-trained VGG16 features
- **Faster Training**: Frozen base layers reduce computation time
- **Better Generalization**: Pre-trained weights provide robust feature extraction
- **Reduced Overfitting**: Lower risk due to pre-trained representations

## 📈 Usage Example

```python
# 1. Prepare datasets
train_data, validation_data = prepare_the_datasets(
    train_datasets='/path/to/training',
    validation_datasets='/path/to/validation',
    batch_size=32,
    image_size=224
)

# 2. Load and configure model
conv_base = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
conv_base.trainable = False

model = Sequential([
    conv_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(4, activation='softmax')
])

# 3. Train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=validation_data, epochs=10)

# 4. Visualize results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

## 🎯 Model Evaluation

### Metrics Tracked
- **Training Accuracy**: Performance on training dataset
- **Validation Accuracy**: Performance on validation dataset
- **Loss**: Categorical crossentropy loss values

### Visualization Tools
- Accuracy comparison plots (Training vs Validation)
- Learning curves for model performance analysis
- Sample image display with predictions

## 📁 Project Structure

```
brain-tumor-classification/
├── main.py                    # Main training script
├── data_preprocessing.py      # Data preparation functions
├── model.py                  # Model architecture definition
├── visualization.py          # Plotting and visualization utilities
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── notebooks/
    └── exploration.ipynb     # Data exploration notebook
```
