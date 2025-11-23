# Building Segmentation with ResNet-UNet

This repository implements a semantic segmentation model for building extraction using a **ResNet-50 + UNet** architecture in TensorFlow / Keras. The model is trained on the **Massachusetts Buildings Dataset** and generates binary building mask predictions.

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation & Inference](#evaluation--inference)
- [Exporting to TensorFlow Lite](#exporting-to-tensorflow-lite)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [Inspiration / References](#inspiration--references)
- [License](#license)

---

## Motivation

Automatically identifying and segmenting buildings from aerial or satellite imagery has many real-world applications, such as:

- Urban planning
- Disaster response
- Mapping and GIS
- Infrastructure monitoring

A UNet-like decoder on top of a strong feature extractor like ResNet-50 provides a balance of semantic richness and spatial accuracy.

---

## Dataset

This project uses the **Massachusetts Buildings Dataset**, which includes:

- Training / Validation / Test image tiles
- Corresponding binary building masks
- A `label_class_dict.csv` file mapping mask RGB values to classes
  - Background
  - Building

---

## Architecture

### 🔹 Encoder

**ResNet-50** backbone pretrained on ImageNet (top removed)

Skip connections taken from:
- `conv1_relu`
- `conv2_block3_out`
- `conv3_block4_out`
- `conv4_block6_out`

### 🔹 Decoder (UNet style)

- `UpSampling2D` + concatenation with encoder features
- Two convolutional layers after each upsampling stage
- Progressive reconstruction to original image size

### 🔹 Output

- `Conv2D(1, activation="sigmoid")`
- Predicts a **binary mask** (building vs background)

---

## Preprocessing

- Images loaded with OpenCV (`cv2.imread`)
- Resized to **256 × 256**
- Preprocessed using `tf.keras.applications.resnet.preprocess_input`
- Masks converted to **one-hot encoded** format

### Helper Functions

- `one_hot_encode(label, label_values)` - Converts class labels to one-hot representation
- `reverse_one_hot(one_hot)` - Converts one-hot encoding back to class indices
- `colour_code_segmentation(indices, label_values)` - Maps class indices to RGB colors

Optional data augmentation:
- Random zoom

---

## Training

Training is handled by a custom Python generator:

```python
model.fit(
    train_generator,
    steps_per_epoch=...,
    validation_data=val_generator,
    validation_steps=...,
    epochs=50,
    callbacks=[ModelCheckpoint(...)]
)
```

Key training details:
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Metrics: IoU, Dice Coefficient
- Early stopping and model checkpointing recommended

---

## Evaluation & Inference

### Metrics

The model is evaluated using:
- **Intersection over Union (IoU)** - measures overlap between predicted and ground truth masks
- **Dice Coefficient** - harmonic mean of precision and recall
- **Pixel Accuracy** - overall classification accuracy

### Inference

```python
# Load trained model
model = tf.keras.models.load_model('path/to/model.h5')

# Preprocess input image
image = cv2.imread('image.tif')
image = cv2.resize(image, (256, 256))
image = tf.keras.applications.resnet.preprocess_input(image)
image = np.expand_dims(image, axis=0)

# Generate prediction
prediction = model.predict(image)
mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
```

---

## Exporting to TensorFlow Lite

Convert the trained model to TensorFlow Lite format for edge deployment:

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## Usage

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd building-segmentation-resnet-unet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py --dataset_path /path/to/dataset --epochs 50 --batch_size 32
```

### Inference

```bash
python inference.py --model_path model.h5 --image_path test_image.tif --output_path output_mask.tif
```

---

## Directory Structure

```
building-segmentation-resnet-unet/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── models/
│   └── resnet_unet.py
├── utils/
│   ├── preprocessing.py
│   └── metrics.py
├── train.py
├── inference.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dependencies

- Python 3.7+
- TensorFlow >= 2.5
- Keras (included in TensorFlow)
- NumPy
- OpenCV (`cv2`)
- Matplotlib
- Scikit-learn

Install all dependencies:
```bash
pip install tensorflow>=2.5 opencv-python numpy matplotlib scikit-learn
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## Inspiration / References

This project was inspired by and builds upon existing work in semantic segmentation:

- **UNet for Building Segmentation (PyTorch)**: [Kaggle - UNet for Building Segmentation](https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch)
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition
- Massachusetts Buildings Dataset: [University of Massachusetts - Computer Vision Lab](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.