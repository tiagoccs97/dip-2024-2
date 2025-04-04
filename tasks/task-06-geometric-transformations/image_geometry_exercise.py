# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""
import numpy as np

def _translate_image(img: np.ndarray, intensity=4) -> np.ndarray:
    
    height, width = img.shape
    dy = height // intensity
    dx = width // intensity

    translated_img = np.zeros_like(img)

    src_h_slice = slice(0, height - dy)
    src_w_slice = slice(0, width - dx)
    dst_h_slice = slice(dy, height)
    dst_w_slice = slice(dx, width)
    
    if height > dy and width > dx:
        translated_img[dst_h_slice, dst_w_slice] = img[src_h_slice, src_w_slice]
    return translated_img

def _rotate_image_90_cw(img: np.ndarray) -> np.ndarray:
    
    rotated_img = np.rot90(img, k=-1)
    
    return rotated_img

def _stretch_horizontal(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    
    height, width = img.shape
    new_width = int(np.round(width * factor))

    stretched_coords_col = np.arange(new_width)

    original_coords_col_float = stretched_coords_col / factor

    original_coords_col_int = np.round(original_coords_col_float).astype(int)

    original_coords_col_clipped = np.clip(original_coords_col_int, 0, width - 1)

    stretched_img = img[:, original_coords_col_clipped]
    
    return stretched_img

def _mirror_horizontal(img: np.ndarray) -> np.ndarray:
    
    mirrored_img = img[:, ::-1]
    
    return mirrored_img

def _apply_barrel_distortion(img: np.ndarray, k: float = 5e-6) -> np.ndarray:

    height, width = img.shape
    distorted_img = np.zeros_like(img)

    dest_rows, dest_cols = np.indices(img.shape)

    center_y, center_x = height / 2, width / 2

    x_dest_rel = dest_cols - center_x
    y_dest_rel = dest_rows - center_y

    radius_dest_sq = x_dest_rel**2 + y_dest_rel**2

    scale_factor = 1 / (1 + k * radius_dest_sq)

    x_src_rel = x_dest_rel * scale_factor
    y_src_rel = y_dest_rel * scale_factor

    src_cols_float = x_src_rel + center_x
    src_rows_float = y_src_rel + center_y

    src_cols_int = np.round(src_cols_float).astype(int)
    src_rows_int = np.round(src_rows_float).astype(int)

    valid_mask = (src_rows_int >= 0) & (src_rows_int < height) & \
                 (src_cols_int >= 0) & (src_cols_int < width)

    valid_dest_rows = dest_rows[valid_mask]
    valid_dest_cols = dest_cols[valid_mask]
    valid_src_rows = src_rows_int[valid_mask]
    valid_src_cols = src_cols_int[valid_mask]

    distorted_img[valid_dest_rows, valid_dest_cols] = img[valid_src_rows, valid_src_cols]
    return distorted_img

def apply_geometric_transformations(img: np.ndarray) -> dict:

    if img.ndim != 2:
        raise ValueError("Input image must be a 2D NumPy array (grayscale).")

    results = {
        "translated": _translate_image(img),
        "rotated": _rotate_image_90_cw(img),
        "stretched": _stretch_horizontal(img, factor=1.5),
        "mirrored": _mirror_horizontal(img),
        "distorted": _apply_barrel_distortion(img, k=5e-6) # Adjusted k based on previous example
    }

    return results
