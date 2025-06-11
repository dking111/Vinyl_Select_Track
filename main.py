import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
import math

def get_sample(centre_x, centre_y, r, grey, angle_rad):
    sample = []
    coords = []
    for i in range(r):
        x = int(centre_x + i * math.cos(angle_rad))
        y = int(centre_y + i * math.sin(angle_rad))
        if 0 <= x < grey.shape[1] and 0 <= y < grey.shape[0]:
            sample.append(grey[y, x])
            coords.append((x, y))
    return np.array(sample, dtype=np.float32), coords

def main(verbose=True):
    # Capture a frame from the camera
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image from camera.")
        return

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape
    max_possible_radius = min(width, height) // 2

    # Detect circles
    circles = cv2.HoughCircles(
        grey, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1000,
        param1=100, param2=50,
        minRadius=int(0.4 * max_possible_radius),
        maxRadius=int(0.95 * max_possible_radius)
    )

    if circles is None:
        print("No circle was detected.")
        return

    # Use the largest detected circle
    circles = np.round(circles[0, :]).astype("int")
    x, y, r = max(circles, key=lambda c: c[2])

    # Prepare image for drawing
    color_output = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_output, (x, y), r, (0, 255, 0), 2)  # main circle
    cv2.circle(color_output, (x, y), 3, (255, 0, 0), -1) # center dot

    all_refined_peaks_coords = []

    # Sample in 4 directions (0°, 90°, 180°, 270°)
    for i in range(4):
        angle_rad = (i / 2) * math.pi
        sample, coords = get_sample(x, y, r, grey, angle_rad)

        gradient = np.abs(np.diff(sample))
        peak_indices, _ = find_peaks(gradient, prominence=5,distance=20)

        # Refine peaks: keep only those with next sample intensity ≤ 50
        refined_peaks = [p for p in peak_indices if (p + 1 < len(sample)) and sample[p + 1] <= 50]

        # Store and draw refined peak points
        peak_coords = [coords[p + 1] for p in refined_peaks]
        all_refined_peaks_coords.append(peak_coords)

        for px, py in peak_coords:
            cv2.circle(color_output, (px, py), 2, (0, 0, 255), -1)  # small red dot

        if verbose:
            print(f"Direction {i * 90}°:")
            print(f"  Peaks (indexes): {refined_peaks}")
            print(f"  Peak pixel coords: {peak_coords}")

    # Show final image with peaks
    plt.figure(figsize=(8, 8))
    plt.title("Detected Circle + Groove Peak Dots")
    plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect vinyl grooves from camera image.")
    parser.add_argument('--no-verbose', action='store_false', dest='verbose', help='Disable verbose output')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    main(verbose=args.verbose)
