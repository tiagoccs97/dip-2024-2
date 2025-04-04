# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def MSE(img1, img2):
    return np.mean((img1-img2)**2)
    
def PSNR(img1, img2):
    mse = MSE(img1, img2)
    return 10 * np.log10(1 / mse)

def SSIM(img1, img2):
    mean1, mean2 = np.mean(img1), np.mean(img2)
    var1, var2 = np.var(img1), np.var(img2)
    cov = np.mean((img1 - mean1) * (img2 - mean2))
    c1, c2 = 0.01**2, 0.03**2  # Pequenas constantes para estabilidade
    return ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))

def NPCC(img1, img2):
    mean1, mean2 = np.mean(img1), np.mean(img2)
    return np.mean((img1 - mean1) * (img2 - mean2))/(np.sqrt(np.var(img1))*np.sqrt(np.var(img2)))

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    # Your implementation here
    return {
        "MSE":MSE(i1, i2),
        "PSNR":PSNR(i1, i2),
        "SSIM":SSIM(i1, i2),
        "NPCC": NPCC(i1, i2)
    }
