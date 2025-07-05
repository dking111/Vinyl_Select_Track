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


def find_repeated_similar_distances(distance_lists, tolerance=3.0, min_occurrences=2):
    all_values = []
    for i, sublist in enumerate(distance_lists):
        for val in sublist:
            all_values.append((val, i))  # (value, list_index)

    merged = []
    used_indices = set()

    for i, (val_i, idx_i) in enumerate(all_values):
        if i in used_indices:
            continue
        group = [(val_i, idx_i)]
        matched_indices = {idx_i}

        for j in range(i + 1, len(all_values)):
            val_j, idx_j = all_values[j]
            if abs(val_i - val_j) <= tolerance and idx_j not in matched_indices:
                group.append((val_j, idx_j))
                matched_indices.add(idx_j)
                used_indices.add(j)

        if len(matched_indices) >= min_occurrences:
            values_only = [v for v, _ in group]
            merged.append(np.mean(values_only))

    return sorted(merged)


def main(verbose=True, test_img=False):
    # Load image
    if not test_img:
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture image from camera.")
            return
    else:
        image = cv2.imread("vinyl.jpg", cv2.IMREAD_COLOR)

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape
    max_possible_radius = min(width, height) // 2

    # Detect main circle
    circles = cv2.HoughCircles(
        grey, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1000,
        param1=100, param2=50,
        minRadius=int(0.4 * max_possible_radius),
        maxRadius=int(0.95 * max_possible_radius)
    )

    if circles is None:
        print("No circle was detected.")
        return

    circles = np.round(circles[0, :]).astype("int")
    x, y, r = max(circles, key=lambda c: c[2])

    color_output = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_output, (x, y), r, (0, 255, 0), 2)
    cv2.circle(color_output, (x, y), 3, (255, 0, 0), -1)

    all_refined_peaks_coords = []

    # Sample in 8 directions (every 45 degrees)
    for i in range(8):
        angle_rad = (i / 4) * math.pi
        sample, coords = get_sample(x, y, r, grey, angle_rad)
        gradient = np.abs(np.diff(sample))
        peak_indices, _ = find_peaks(gradient, prominence=5, distance=20)
        refined_peaks = [p for p in peak_indices if (p + 1 < len(sample)) and sample[p + 1] <= 50]
        peak_coords = [coords[p + 1] for p in refined_peaks]
        all_refined_peaks_coords.append(peak_coords)

        if verbose:
            print(f"Direction {int(i * 45)}°:")
            print(f"  Peaks (indexes): {refined_peaks}")
            print(f"  Peak pixel coords: {peak_coords}")

    # Convert peak coords to distances (mm)
    LP_R_mm = 302 / 2  # Vinyl radius in mm
    distances = []

    for peak_coords in all_refined_peaks_coords:
        dir_distances = []
        for px, py in peak_coords:
            dist_px = math.sqrt((px - x) ** 2 + (py - y) ** 2)
            dist_mm = dist_px * (LP_R_mm / r)
            distance_from_edge_mm = LP_R_mm - dist_mm
            dir_distances.append(distance_from_edge_mm)
        distances.append(dir_distances)

    # Merge distances across directions
    final_distances = find_repeated_similar_distances(distances)

    # Draw full groove circles for final distances
    for d in final_distances:
        groove_radius_mm = LP_R_mm - d
        groove_radius_px = groove_radius_mm * (r / LP_R_mm)
        cv2.ellipse(
            color_output,
            (x, y),  # center
            (int(groove_radius_px), int(groove_radius_px)),  # axes (rx, ry)
            0,       # rotation angle
            0, 180,  # start and end angle (draw upper half, 0° to 180°)
            (255, 0, 255),  # magenta color
            1        # thickness
        )

    # Draw all detected peak points
    for peak_coords in all_refined_peaks_coords:
        for px, py in peak_coords:
            cv2.circle(color_output, (px, py), 2, (0, 0, 255), -1)  # red



    track_length = []
    track_elapsed_times = list(map(lambda x: (x / 86) * 1200, final_distances))
    print(f"Track_Elpased_Times: {track_elapsed_times}")

    for i in range(len(track_elapsed_times)):
        if i == 0:
            track_length.append(track_elapsed_times[i])
        else:
            track_length.append(track_elapsed_times[i] - track_elapsed_times[i - 1])

    


    

    # Show final visualization
    plt.figure(figsize=(8, 8))
    plt.title("Detected Circle + Groove Peak Dots")
    plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect vinyl grooves from camera image.")
    parser.add_argument('--no-verbose', action='store_false', dest='verbose', help='Disable verbose output')
    parser.add_argument('--test-img', action='store_true', dest='test_img', help='Use a test image instead of camera')
    parser.set_defaults(verbose=True)
    parser.set_defaults(test_img=False)
    args = parser.parse_args()
    main(verbose=args.verbose, test_img=args.test_img)
