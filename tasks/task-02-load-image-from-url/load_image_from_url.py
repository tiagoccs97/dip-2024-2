import requests
import cv2 as cv
import numpy as np

def load_image_from_url(url, **kwargs):
    # Check for any invalid keyword arguments
    valid_kwargs = {'flags'}
    invalid_keys = set(kwargs.keys()) - valid_kwargs
    if invalid_keys:
        raise TypeError(f"Invalid keyword argument(s): {', '.join(invalid_keys)}")
    
    # Set default flags if not provided
    flags = kwargs.get('flags', cv.IMREAD_COLOR)
    
    # Download the image from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Convert the response content to a numpy array
    image_array = np.frombuffer(response.content, dtype=np.uint8)
    
    # Decode the image using OpenCV
    image = cv.imdecode(image_array, flags)
    
    return image
