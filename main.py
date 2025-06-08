import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse

def main(verbose=True, graphical=True):
    # --- Capture a frame from the camera ---
    cap = cv2.VideoCapture(0)  # Try 0 for default webcam
    ret, image = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image from camera.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    max_possible_radius = min(width, height) // 2

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1000,
        param1=100, param2=50,
        minRadius=int(0.4 * max_possible_radius),
        maxRadius=int(0.95 * max_possible_radius)
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = max(circles, key=lambda c: c[2])

        color_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(color_output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(color_output, (x, y), 3, (255, 0, 0), -1)

        vert_line = []
        for radius in range(r):
            yy = y + radius
            if 0 <= yy < gray.shape[0]:
                vert_line.append(gray[yy, x])

        vert_line = np.array(vert_line, dtype=np.float32)
        gradient = np.abs(np.diff(vert_line))
        peaks, _ = find_peaks(gradient, prominence=5)

        refined_peaks = [p for p in peaks if (p + 1 < len(vert_line)) and vert_line[p + 1] <= 50]
        refined_peaks = np.array(refined_peaks)

        for p in refined_peaks:
            cv2.circle(color_output, (x, y), p, (200, 200, 200), 3)

        for p in refined_peaks:
            px, py = x, (p + 1) + y
            cv2.circle(color_output, (px, py), 2, (0, 0, 255), -1)

        scale_factor = 152.4 / r
        distance_from_outer_edge = list(map(lambda x: round((r - x) * scale_factor, 2), refined_peaks))

        if verbose:
            print(f"Refined peaks: {refined_peaks}")
            print(f"Distance to outer edge (mm): {distance_from_outer_edge}")

        if graphical:
            plt.figure(figsize=(15, 6))

            plt.subplot(1, 3, 1)
            plt.title("Detected Circle + Track Edges")
            plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Vertical Intensity Profile")
            plt.plot(vert_line, label="Intensity")
            plt.plot(refined_peaks + 1, vert_line[refined_peaks + 1], "rx", label="Detected Edges")
            plt.xlabel("Pixels from center outward")
            plt.ylabel("Intensity")
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.title("Line Visualization (Gray)")
            line_img = vert_line.astype(np.uint8).reshape(-1, 1)
            plt.imshow(line_img, cmap='gray', aspect='auto')
            plt.axis("off")

            plt.tight_layout()
            plt.show()
    else:
        print("No circle was detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect vinyl grooves from camera image.")
    parser.add_argument('--no-verbose', action='store_false', dest='verbose', help='Disable verbose output')
    parser.add_argument('--no-graphical', action='store_false', dest='graphical', help='Disable graphical display')
    parser.set_defaults(verbose=True, graphical=True)
    args = parser.parse_args()

    main(verbose=args.verbose, graphical=args.graphical)
