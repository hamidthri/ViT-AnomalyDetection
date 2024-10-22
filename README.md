# ViT Autoencoder for Image Reconstruction

This repository contains a Vision Transformer (ViT) based autoencoder for image reconstruction. The architecture leverages a pre-trained Vision Transformer as the encoder, with a custom decoder to reconstruct input images. The project includes functionality for both training and testing the autoencoder, with a focus on patch-wise error analysis to highlight anomalies in reconstructed images.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Model Architecture](#model-architecture)
- [Results](#results)
  - [Reconstructed Images and Differences](#reconstructed-images-and-differences)
  - [Patch-wise Error Analysis](#patch-wise-error-analysis)



## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/hamidthri/ViT-AnomalyDetection
    cd vit-autoencoder
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install `timm` for Vision Transformer models:
    ```bash
    pip install timm
    ```

4. Set up the dataset structure:
   - Make sure your dataset is structured as follows:
     ```plaintext
     dataSet/
       ├── classified_kraftig/
           ├── train/
               |── good
           ├── test/
               |── good
               |── anomalous
     ```

## Usage

### Training

To train the autoencoder, run the following command:

```bash
python vit_autoencoder.py --mode train --checkpoint <path_to_checkpoint> 
```
--mode: Specifies the mode (train or test).
--checkpoint: (Optional) Path to a saved checkpoint to resume training from.
### Testing
To test the model with patch-wise error analysis:

```
python vit_autoencoder.py --mode test --checkpoint <path_to_checkpoint>
```
--mode: test for testing the model.
--checkpoint: Path to the trained model checkpoint.



## Model Architecture

The autoencoder in this project has two main components:

- **Encoder**: A pre-trained Vision Transformer (ViT) that captures high-level features from the input images. The ViT encoder is initialized with weights pre-trained on ImageNet to improve feature extraction and reduce training time.
  
- **Decoder**: A custom-designed convolutional decoder that reconstructs images from the encoded features. It uses several transpose convolutional layers, each with instance normalization and ReLU activation, to gradually upsample and restore the input image's dimensions.

### Key Features:
- **Pre-trained Transformer Encoder**: Utilizes a state-of-the-art Vision Transformer for image feature encoding.
- **Custom Decoder**: A fully convolutional decoder for precise image reconstruction.
- **Patch-wise Error Analysis**: Computes reconstruction error on image patches to detect fine-grained anomalies.
- **Checkpointing**: Model training can be resumed or tested with saved checkpoints.

## Results

### Reconstructed Images and Differences

During testing, the autoencoder outputs:

1. **Original Image**: The input image fed to the model.
2. **Reconstructed Image**: The image reconstructed by the model.
3. **Difference Image**: Pixel-wise absolute difference between the original and reconstructed images, highlighting areas where reconstruction errors occur.

### Patch-wise Error Analysis

The model also performs patch-wise error analysis to detect and localize anomalies:

- The input image is divided into patches.
- Mean Squared Error (MSE) is calculated for each patch between the input and reconstructed images.
- A patch-wise error map is generated to visually represent regions with high reconstruction error, which can indicate anomalies.

Example of results:
- **Original Image**
- **Reconstructed Image**
- **Difference Image (highlighting reconstruction errors)**
- **Patch-wise Error Map (showing anomaly regions)**

These visualizations help to interpret model performance and identify regions where the model struggles to accurately reconstruct the image.

