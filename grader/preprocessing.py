"""
Card Normalization & Preprocessing Pipeline
Detects card boundaries, applies perspective correction, resizes to standard dimensions.
"""

import cv2
import numpy as np

# Standard card dimensions (poker card ratio ~1:1.4)
CARD_WIDTH = 500
CARD_HEIGHT = 700


def load_image(file_bytes: bytes) -> np.ndarray:
    """Load image from raw bytes into BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def load_image_from_path(path: str) -> np.ndarray:
    """Load image from file path."""
    return cv2.imread(path)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    Required for consistent perspective transform.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest diff
    return rect


def four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective warp to achieve top-down view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return warped


def detect_card_contour(img: np.ndarray) -> np.ndarray:
    """
    Find the card boundary contour in the image.
    Returns the best 4-point approximation or None.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try multiple preprocessing approaches for robustness
    candidates = []

    # Approach 1: Canny with bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(filtered, 30, 120)
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
    candidates.append(edged)

    # Approach 2: Adaptive threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    candidates.append(thresh)

    best_contour = None
    best_area = 0

    for edge_map in candidates:
        contours, _ = cv2.findContours(edge_map.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Sort by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            # Must be at least 20% of image area to be the card
            if area < (h * w * 0.20):
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4 and area > best_area:
                best_area = area
                best_contour = approx
                break

    return best_contour


def normalize_card(img: np.ndarray) -> tuple:
    """
    Full normalization pipeline:
    1. Detect card boundaries
    2. Apply perspective transform
    3. Resize to standard dimensions (500x700)

    Returns:
        (normalized_img, detection_succeeded: bool)
    """
    if img is None:
        raise ValueError("Input image is None")

    h, w = img.shape[:2]

    # Attempt contour-based detection
    contour = detect_card_contour(img)

    if contour is not None and len(contour) == 4:
        pts = contour.reshape(4, 2).astype("float32")
        try:
            warped = four_point_transform(img, pts)
            # Validate warped dimensions make sense (aspect ratio sanity check)
            wh, ww = warped.shape[:2]
            if ww > 0 and 0.5 < (wh / ww) < 3.0:
                resized = cv2.resize(warped, (CARD_WIDTH, CARD_HEIGHT),
                                     interpolation=cv2.INTER_LANCZOS4)
                return resized, True
        except Exception:
            pass

    # Fallback: crop to largest bounding rect and resize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        # Only crop if the bounding box is meaningful
        if bw > w * 0.3 and bh > h * 0.3:
            cropped = img[y:y + bh, x:x + bw]
            resized = cv2.resize(cropped, (CARD_WIDTH, CARD_HEIGHT),
                                 interpolation=cv2.INTER_LANCZOS4)
            return resized, False

    # Last resort: just resize the whole image
    resized = cv2.resize(img, (CARD_WIDTH, CARD_HEIGHT),
                         interpolation=cv2.INTER_LANCZOS4)
    return resized, False
