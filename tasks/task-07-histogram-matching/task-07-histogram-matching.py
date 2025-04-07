# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
#import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    # Your implementation here
    matched_img = np.zeros_like(source_img)
    
    for canal in range(3):  #R, G, B

        fonte = source_img[:,:,canal]
        ref = reference_img[:,:,canal]
        
        hist_fonte = np.histogram(fonte, bins=256, range=(0,255))[0]
        hist_ref = np.histogram(ref, bins=256, range=(0,255))[0]
        
        print("fonte\n", fonte, "\n", "hist_fonte\n", hist_fonte)
        
        cdf_fonte = hist_fonte.cumsum()
        cdf_fonte = cdf_fonte / cdf_fonte[-1]
        
        cdf_ref = hist_ref.cumsum()
        cdf_ref = cdf_ref / cdf_ref[-1]
        
        mapeamento = np.zeros(256, dtype=np.uint8)
        for v in range(256):
            mapeamento[v] = np.argmin(np.abs(cdf_ref - cdf_fonte[v]))
        
        matched_img[:,:,canal] = mapeamento[fonte]
    
    return matched_img
