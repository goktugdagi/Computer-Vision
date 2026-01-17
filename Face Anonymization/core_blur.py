import cv2                     # OpenCV for image processing
import numpy as np              # NumPy for array operations


def build_faces_mask(frame_shape, all_faces_landmarks):
    # Extract frame height and width
    h, w = frame_shape[:2]

    # Create an empty black mask
    mask = np.zeros((h, w), np.uint8)

    # Iterate over landmarks of each detected face
    for landmarks in all_faces_landmarks:
        # Skip invalid or too-small landmark sets
        if landmarks is None or len(landmarks) < 3:
            continue

        # Compute convex hull around facial landmarks
        hull = cv2.convexHull(landmarks)

        # Fill the convex hull area on the mask
        cv2.fillConvexPoly(mask, hull, 255)

    # Return combined mask for all faces
    return mask


def _adaptive_kernel(all_faces_landmarks, base_ksize=27, max_ksize=99):
    # Start with base kernel size
    k = int(base_ksize)

    # Ensure kernel size is at least 1
    if k < 1:
        k = 1

    # If no faces detected, just return base kernel
    if not all_faces_landmarks:
        if k % 2 == 0:
            k += 1
        return k

    # Stack all landmark points into one array
    pts = np.vstack(all_faces_landmarks)

    # Compute bounding rectangle of all faces
    x, y, w, h = cv2.boundingRect(pts)

    # Suggest blur size proportional to face size
    suggested = int(0.22 * min(w, h))

    # Use the larger of base or suggested kernel
    k = max(k, suggested)

    # Clamp kernel size to maximum allowed value
    k = min(k, max_ksize)

    # Kernel size must be odd for GaussianBlur
    if k % 2 == 0:
        k += 1

    return k


def _dilate_mask(mask, k):
    """
    Slightly expand the mask to avoid sharp edges around blurred faces.
    """

    # Determine dilation kernel size based on blur kernel
    dil = max(5, min(31, k // 4))

    # Ensure dilation kernel size is odd
    if dil % 2 == 0:
        dil += 1

    # Create elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))

    # Dilate the mask once
    return cv2.dilate(mask, kernel, iterations=1)


def blur_faces(frame, all_faces_landmarks, blur_ksize=27):
    # If frame is invalid, return None
    if frame is None:
        return None

    # Build mask covering all detected faces
    mask = build_faces_mask(frame.shape, all_faces_landmarks)

    # If mask is empty, return original frame
    if mask.max() == 0:
        return frame

    # Compute adaptive blur kernel size
    k = _adaptive_kernel(all_faces_landmarks, base_ksize=blur_ksize, max_ksize=99)

    # Expand mask to improve blur coverage
    mask = _dilate_mask(mask, k)

    # Apply Gaussian blur to entire frame
    frame_blur = cv2.GaussianBlur(frame, (k, k), 0)

    # Extract blurred face regions
    faces_blurred = cv2.bitwise_and(frame_blur, frame_blur, mask=mask)

    # Extract unblurred background
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    # Combine blurred faces with original background
    return cv2.add(background, faces_blurred)
