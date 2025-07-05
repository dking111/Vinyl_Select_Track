import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
import math
from typing import Tuple,List
TEST_IMG_PTH = "vinyl.jpg"
NUM_SAMPLES = 8
LP_R = 302 / 2 



def get_sample(centre_x: np.int32, centre_y: np.int32, r: np.int32,
               grey: np.ndarray, angle_rad: float) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Gets the pixel values along a line from the centre to the edge of the vinyl.

    Args:
        centre_x    (np.int32)  : The x-coordinate of the vinyl's centre.
        centre_y    (np.int32)  : The y-coordinate of the vinyl's centre.
        r           (np.int32)  : The radius (in pixels) of the detected vinyl.
        grey        (np.ndarray): The greyscale image.
        angle_rad   (float)     : The angle (in radians) at which the sample line is taken.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]]]: 
            A np array of sampled greyscale values and a list of their (x, y) coordinates.
    """
    sample = []
    coords = []
    #iterate through calculated pixel positions
    for i in range(r):
        x = int(centre_x + i * math.cos(angle_rad))
        y = int(centre_y + i * math.sin(angle_rad))
        if 0 <= x < grey.shape[1] and 0 <= y < grey.shape[0]:
            sample.append(grey[y, x])
            coords.append((x, y))
    return np.array(sample, dtype=np.float32), coords


def find_repeated_similar_distances(distance_lists:List[List[np.float64]], tolerance:float = 3.0, min_occurrences:int = 2) -> List[np.float64]:
    """
    Takes all recorded distances and filters them such that similar and repeated distances are ommitted.

    Args:
        distance_lists  (List[List[np.float64]]): A 2D array of all distance data                                       TODO: Convert to np array
        tolerance       (float)                 : The distance between points required to be considered distinct
        min_occurences: (int)                   : The number of occurences required for a point to be added to final set

    Returns:
        List[np.float64]                        : The final set of distinct distances
    """
    #Flattens the list into a 1D array of tuples
    all_values = []
    for i, sublist in enumerate(distance_lists):
        for val in sublist:
            all_values.append((val, i)) 

    merged = []
    used_indices = set()
    #Group similar values from different sources
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
        #Keep groups that occur enough, takes mean
        if len(matched_indices) >= min_occurrences:
            values_only = [v for v, _ in group]
            merged.append(np.mean(values_only))
    #Return sorted list
    return sorted(merged)


def load_grey_img(test_img:bool = False) -> np.ndarray:
    """
    Loads an image from the camera or a test image, and converts it to greyscale.

    Args:
        test_img (bool):    If True, loads a predefined test image. If False, captures from the camera.

    Returns:
        np.ndarray:         The greyscale image.

    Raises:
        RuntimeError:       If the camera doesn't exist or can't be accessed
        FileNotFoundEror:   If the test image file cannot be loaded
    """
    #Load camera image
    if not test_img:
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture image from camera.")
    #Load test image 
    else:
        image = cv2.imread(TEST_IMG_PTH, cv2.IMREAD_COLOR)
        if type(image) == None:
            raise FileNotFoundError(f"Test image {TEST_IMG_PTH} couldn't be found")
    #convert and return
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey



def get_largest_circle(grey:np.ndarray, max_r:int) -> Tuple[np.int32, np.int32, np.int32]:
    """
    Given an image, detects all possible circles using Hough Circles Transform and returns the largest.

    Args:
        grey    (np.ndarray): The greyscale image.
        max_r   (int)       : The largest possible radius (half the size of the image)

    Returns:
        Tuple[np.int32, np.int32, np.int32]:
            The coordinate of the centre of the largest circle and its radius

    Raises:
        RuntimeError: If no circles are detected within the image
    """
    #Find circle candidates using Hough Circles Transform
    circles = cv2.HoughCircles(
        grey, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1000,
        param1=100, param2=50,
        minRadius=int(0.7 * max_r),
        maxRadius=int(1.00 * max_r)
    )

    if circles is None:
        raise RuntimeError("No circles were detected")

    #Return circle with largest radius
    circles = np.round(circles[0, :]).astype("int")
    x, y, r = max(circles, key=lambda c: c[2])
    return (x,y,r)



def get_and_refine_peaks(sample:np.ndarray, coords:List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """
    Finds a list of refined peaks from a sample.

    Args:
        sample (np.ndarray): A 1D array of sampled values (np.float32)
        coords (List[Tuple[int][int]]): The coordinates corresponding to the sampled values

    Returns: 
        List[Tuple[int][int]]:
            The coordinates of the refined peaks
    """
    #Finds gradient 
    gradient = np.abs(np.diff(sample))
    #Finds peaks in gradient
    peak_indices, _ = find_peaks(gradient, prominence=5, distance=20)
    #Refines peaks based on cutoff
    refined_peaks = [p for p in peak_indices if (p + 1 < len(sample)) and sample[p + 1] <= 50]
    #Maps peaks to a set of coords
    peak_coords = [coords[p + 1] for p in refined_peaks]
    return peak_coords


def coords_to_distances(x,y,r,coords):
    """
    Takes the 2D array containing the peak coordinates for each sample and converts it distance from the centre of the circle.

    Args:
        x       (np.float32): The x coordinate of the centre of the largest circle.
        y       (np.float32): The y coordinate of the centre of the largest circle.
        r       (np.float32): The radius of the largest circle.
        coords  (np.ndarray): The 2D array containing the coordinates (np.float32,np.float32) of each sample.

    Returns:
        List[List[np.float64]]:
            A 2D array of the distances from the centre for each sample. TODO: Make np.float32 ndarray
    """
    distances = []
    #iteratates through each coordinate and applies forumula to calc distance from centre
    for peak_coords in coords:
        dir_distances = []
        for px, py in peak_coords:
            #formula
            dist_px = math.sqrt((px - x) ** 2 + (py - y) ** 2)
            dist_mm = dist_px * (LP_R / r)
            distance_from_edge_mm = LP_R - dist_mm
            dir_distances.append(distance_from_edge_mm)
        distances.append(dir_distances)
    return distances


def create_graph(x,y,r,grey,final_distances,all_refined_peaks_coords):
    """
    Creates a visual output of where all data including: vinyl centre, vinyl cirumference, detected grooves, refined grooves

    Args:
        x                           (np.float32)        : The x coordinate of the centre of the largest circle.
        y                           (np.float32)        : The y coordinate of the centre of the largest circle.
        r                           (np.float32)        : The radius of the largest circle.
        grey                        (np.ndarray)        : The greyscale image.
        final_distances             (List[np.float64])  : The final distances of each detected groove.
        all_refined_peaks_coords    (np.ndarray)        : 2D array of the refined peaks in each sample.
    
    Returns:
        None
    """
    #Center and circumference
    color_output = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_output, (x, y), r, (0, 255, 0), 2)
    cv2.circle(color_output, (x, y), 3, (255, 0, 0), -1)
    
    #All detected peak points
    for peak_coords in all_refined_peaks_coords:
        for px, py in peak_coords:
            cv2.circle(color_output, (px, py), 2, (0, 0, 255), -1)  

    #Groove circles for final distances
    for d in final_distances:
        groove_radius_mm = LP_R - d
        groove_radius_px = groove_radius_mm * (r / LP_R)
        cv2.ellipse(color_output, (x, y), (int(groove_radius_px), int(groove_radius_px)), 0, 0, 180, (255, 0, 255), 1)

    #Create final graph
    plt.figure(figsize=(8, 8))
    plt.title("Detected Circle + Groove Peak Dots")
    plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()



def get_track_lengths(final_distances):
    """
    Calculates the individual track lengths based on the distances of each groove.

    Args:
        final_distances (List[np.float64]): The distances (mm) from the centre of the vinyl of each groove.

    Returns:
        List[np.float64]: 
            A list of the length of each track.
    """
    track_lengths = []
    track_elapsed_times = list(map(lambda x: (x / 86) * 1200, final_distances))
    for i in range(len(track_elapsed_times)):
        if i == 0:
            track_lengths.append(track_elapsed_times[i])
        else:
            track_lengths.append(track_elapsed_times[i] - track_elapsed_times[i - 1])
    return track_lengths



def main(verbose=True, test_img=False):
    #load img
    grey = load_grey_img(test_img)
    #extract useful info
    height, width = grey.shape
    max_r = min(width, height) // 2
    #Find vinyl
    x,y,r = get_largest_circle(grey,max_r)
    # Sample in all directions 
    all_refined_peaks_coords = []
    for i in range(NUM_SAMPLES):
        angle_rad = (i / (NUM_SAMPLES/2)) * math.pi
        sample, coords = get_sample(x, y, r, grey, angle_rad)
        peak_coords = get_and_refine_peaks(sample,coords)
        all_refined_peaks_coords.append(peak_coords)

    #convert from real world distances
    distances = coords_to_distances(x,y,r,all_refined_peaks_coords)
    # Merge distances across directions
    final_distances = find_repeated_similar_distances(distances)
    #Calc useful data
    track_lengths = get_track_lengths(final_distances)
    #visualise
    create_graph(x,y,r,grey,final_distances,all_refined_peaks_coords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect vinyl grooves from camera image.")
    parser.add_argument('--no-verbose', action='store_false', dest='verbose', help='Disable verbose output')
    parser.add_argument('--test-img', action='store_true', dest='test_img', help='Use a test image instead of camera')
    parser.set_defaults(verbose=True)
    parser.set_defaults(test_img=False)
    args = parser.parse_args()
    main(verbose=args.verbose, test_img=args.test_img)
