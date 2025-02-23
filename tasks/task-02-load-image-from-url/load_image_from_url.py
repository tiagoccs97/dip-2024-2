import requests
import cv2 as cv
import numpy as np

def load_image_from_url(url, **kwargs):
        """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###
    valid_kwargs = {'flags'}
    invalid_keys = set(kwargs.keys()) - valid_kwargs
    if invalid_keys:
        raise TypeError(f"Invalid keyword argument(s): {', '.join(invalid_keys)}")
        
    flags = kwargs.get('flags', cv.IMREAD_COLOR)    
    response = requests.get(url)
    response.raise_for_status()  
    image_array = np.frombuffer(response.content, dtype=np.uint8)
    image = cv.imdecode(image_array, flags)
    ### END CODE HERE ###
    
    return image
