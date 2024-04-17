#  Facial Image Denoising using U-Net

This project focuses on enhancing the quality of facial images by removing noise using the U-Net architecture.

## Introduction

Denoising facial images is essential for improving image quality in various applications, including image processing and computer vision tasks. This project demonstrates the application of deep learning techniques, specifically the U-Net model, for facial image denoising.

## Methodology

### Data Collection

The dataset utilized for this project comprises noisy facial images and corresponding clean images, facilitating the training of a facial image denoising model. Specifically:

- **Dataset**: FFHQ (Flickr-Faces-HQ) dataset of 128 x 128 pixel-sized facial images.

- **Noise Synthesis**: Synthetic noise is applied to the clean images using a noise factor of 0.3 to create noisy counterparts.

- Data Size

  : A total of 10,000 image pairs are used:

  - 8,000 images for training the model.
  - 20% (1,600 images) reserved for validation during training.
  - 2,000 images for testing the trained model.

This dataset setup ensures a diverse and representative collection for training and evaluating the denoising model effectively.

### Data Preprocessing

- Data augmentation techniques are employed to diversify the training dataset.
- Image normalization and resizing are conducted to prepare the data for training.

### Model Architecture

- **U-Net**: The U-Net architecture is leveraged for image-to-image translation, specifically designed for denoising facial images. U-Net comprises an encoder-decoder network with skip connections, effectively preserving spatial information.

  ![U-Net Architecture](https://github.com/mauluddin12z/FACIAL_IMAGE_DENOISING_UNET/blob/main/architecture_img/unet_architecture.png)

### Training

- The U-Net model is trained on pairs of noisy and clean images using mean squared error (MSE) loss for optimization.

### Evaluation

- The performance of the trained model is assessed using **Peak Signal-to-Noise Ratio (PSNR)** and **Structural Similarity Index (SSIM)** metrics to quantify noise reduction and image quality improvement.

## Dependencies

Ensure you have the following dependencies installed:

- Python (>=3.10)
- TensorFlow (2.10)
- numpy
- OpenCV (for image processing)
- matplotlib (for visualization)

```
bashCopy code
pip install tensorflow==2.10 numpy opencv-python matplotlib
```

## Acknowledgements

- The FFHQ dataset of 128x128 thumbnails used for this project.
