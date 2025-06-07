import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load image
image = cv2.imread("vinyl.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get image dimensions
height, width = gray.shape
max_possible_radius = min(width, height) // 2

# Detect large outer circle (vinyl boundary)
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1000,
    param1=100, param2=50,
    minRadius=int(0.4 * max_possible_radius),
    maxRadius=int(0.95 * max_possible_radius)
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    # Select the largest circle
    x, y, r = max(circles, key=lambda c: c[2])

    # Draw the circle and center
    color_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_output, (x, y), r, (0, 255, 0), 2)
    cv2.circle(color_output, (x, y), 3, (255, 0, 0), -1)

    # Collect vertical intensity values from center outward (along +y)
    vert_line = []
    coords = []
    for radius in range(r):
        yy = y + radius
        if 0 <= yy < gray.shape[0]:
            vert_line.append(gray[yy, x])
            coords.append((x, yy))  # Store pixel locations for peak drawing

    # Convert to float for processing
    vert_line = np.array(vert_line, dtype=np.float32)

    # 1D edge detection: gradient magnitude
    gradient = np.abs(np.diff(vert_line))
    peaks, _ = find_peaks(gradient, prominence=5)

    # Filter out peaks where intensity exceeds 100
    refined_peaks = [p for p in peaks if (p + 1 < len(vert_line)) and vert_line[p + 1] <= 50]
    refined_peaks = np.array(refined_peaks)

    # Draw refined peaks on image
    for p in refined_peaks:
        px, py = coords[p + 1]  # +1 due to np.diff()
        cv2.circle(color_output, (px, py), 2, (0, 0, 255), -1)

    # Plot results
    plt.figure(figsize=(15, 6))

    # Show image with circle and detected peaks
    plt.subplot(1, 3, 1)
    plt.title("Detected Circle + Track Edges")
    plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Plot intensity profile with peaks
    plt.subplot(1, 3, 2)
    plt.title("Vertical Intensity Profile")
    plt.plot(vert_line, label="Intensity")
    plt.plot(refined_peaks + 1, vert_line[refined_peaks + 1], "rx", label="Detected Edges")
    plt.xlabel("Pixels from center outward")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.legend()

    # Show intensity values as vertical grayscale strip
    plt.subplot(1, 3, 3)
    plt.title("Line Visualization (Gray)")
    line_img = vert_line.astype(np.uint8).reshape(-1, 1)
    plt.imshow(line_img, cmap='gray', aspect='auto')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

else:
    print("No circle was detected.")
