import cv2
import numpy as np

def linear_combination(i1: np.ndarray, i2: np.ndarray, a1: float, a2: float) -> np.ndarray:
    """
    Compute the linear combination of two images using OpenCV: 
    i_out = a1 * i1 + a2 * i2.

    Args:
        i1 (np.ndarray): First input image.
        i2 (np.ndarray): Second input image.
        a1 (float): Scalar weight for the first image.
        a2 (float): Scalar weight for the second image.

    Returns:
        np.ndarray: The resulting image with the same dtype as the input images.
    """
    # Ensure images have the same dimensions
    if i1.shape != i2.shape:
        raise ValueError("Input images must have the same dimensions.")

    ### START CODE HERE ###
    if i1 is None:
        print("Error loading i1")
        exit(-1)
    elif i2 is None:
        print("Error loading i2")
        exit(-1)
    # Blending images
    bld = cv2.addWeighted(i1, a1, i2, a2, 0.0)
    # Display
    cv2.imshow('Blended Images', bld)
    cv2.waitKey(0)
    #Display Clear
    cv2.destroyAllWindows()
    ### END CODE HERE ###

    return None

# Example Usage
if __name__ == "__main__":
    # Load images
    i1 = cv2.imread('image1.png', cv2.IMREAD_COLOR)
    i2 = cv2.imread('image2.png', cv2.IMREAD_COLOR)

    if i1 is None or i2 is None:
        raise FileNotFoundError("One or both images could not be loaded. Check file paths.")

    # Define scalars
    a1, a2 = 0.6, 0.4

    # Compute the linear combination
    output = linear_combination(i1, i2, a1, a2)
    
