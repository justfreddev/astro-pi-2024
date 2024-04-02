import cv2 as cv
import math
import numpy as np
from orbit import ISS
import os
from picamera import PiCamera # type: ignore (creates error because its an astro-pi-replay library)
from queue import Queue
import scipy.stats as sps
from scipy.signal import argrelextrema
from skyfield.api import load
from time import time
from typing import Any, Tuple

IMAGE_WIDTH, IMAGE_HEIGHT = 4056, 3040
SENSOR_WIDTH = 6.287
FOCAL_LENGTH = 5

ISS = ISS()
cam = PiCamera()
cam.resolution = (IMAGE_WIDTH, IMAGE_HEIGHT)

current_dir = os.getcwd()
photos_dir = os.path.join(current_dir)

distances = []


def calculate_speed_in_kmps(feature_distance, GSD):
    """Calculate the distance by multiplying the feature distance by the Ground Sample Distance (GSD) and dividing by 100,000"""
    
    # This converts the feature distance from pixels to kilometers
    distance = feature_distance * GSD / 100_000

    # Return the ratio of the feature distance to the calculated distance
    # This is a measure of how much the feature has moved in the image, relative to the distance it represents on the ground
    return feature_distance / distance


def get_GSD(ISS: ISS) -> int: # type: ignore (creates error because its an astro-pi-replay library)
    """Gets the ground sample distance of the ISS"""
    
    # Get the current time in the timescale used by the skyfield library
    t = load.timescale().now()

    # Calculate the altitude of the International Space Station (ISS) at the current time
    # The altitude is the height of the ISS above the Earth's surface, in meters
    altitude = ISS.at(t).subpoint().elevation.m

    # Calculate the Ground Sample Distance (GSD), which is the distance between pixel centers measured on the ground
    # The formula for GSD is (sensor width * altitude) / (focal length * image width)
    # The result is multiplied by 100 to convert from meters to centimeters
    GSD = ((SENSOR_WIDTH * altitude) / (FOCAL_LENGTH * IMAGE_WIDTH)) * 100

    # Return the calculated GSD
    return GSD


def filter_distances(distances: list[float]) -> list[float]:
    """Filters the distances using a gaussian kernel density estimation"""
    
    # Create a Gaussian Kernel Density Estimation (KDE) of the distances
    density: sps.gaussian_kde = sps.gaussian_kde(distances)

    # Generate a sequence of evenly spaced values over the range of distances
    values = np.linspace(min(distances), max(distances), len(distances))

    # Evaluate the KDE for each value in the sequence
    density_values: list[float] = density(values)

    # Multiply each density value by 10000 and convert to integers
    density_values: list[int] = [x * 10000 for x in density_values]

    # Find the maximum density value
    max_density: int = max(density_values)

    # Find the value corresponding to the maximum density
    max_density_value = values[density_values.index(max_density)]

    # Return the value corresponding to the maximum density
    return max_density_value


def calculate_mean_distance(coordinates_1, coordinates_2) -> list[float]:
    """Calculates the mean distance between the coordinates of the two images"""
    
    # Merge the coordinates from the first and second image into a single list of tuples
    merged_coordinates = list(zip(coordinates_1, coordinates_2))

    # Iterate over each pair of coordinates
    for coordinate in merged_coordinates:
        # Calculate the difference in x-coordinates between the first and second image
        x_difference = coordinate[0][0] - coordinate[1][0]
        # Calculate the difference in y-coordinates between the first and second image
        y_difference = coordinate[0][1] - coordinate[1][1]
        # Calculate the Euclidean distance between the pair of coordinates using the Pythagorean theorem
        distance = math.hypot(x_difference, y_difference)
        # Append the calculated distance to the list of distances
        distances.append(distance)

    # Filter the list of distances using a custom filter function
    filtered_distances: list[float] = filter_distances(distances)

    # Return the filtered list of distances
    return filtered_distances


def filter_gradients(gradients: list[float]) -> float:
    """Filters the gradients using a gaussian kernel density estimation and argrelextrema to find the maxima of the kde."""
    
    # Create a Gaussian Kernel Density Estimation (KDE) of the gradients
    kde = sps.gaussian_kde(gradients)

    # Evaluate the KDE for each gradient
    y = kde(gradients)

    # Find the indices of the local maxima in the KDE
    maxima_indices = argrelextrema(y, np.greater)[0]

    # Get the y-value of the first local maximum
    maxima_y = y[maxima_indices[0]]

    # Find the indices of the y-values that are greater than or equal to the first local maximum
    above_threshold_indices = np.where(y >= maxima_y)[0].tolist()

    # Get the gradients corresponding to the above-threshold indices
    x_values_above_threshold = [gradients[i] for i in above_threshold_indices]

    # Return the gradients that are above the threshold
    return x_values_above_threshold


def calculate_gradients(coords_1, coords_2):
    """Calculates the gradients between the coordinates of the two images"""
    
    # Initialize an empty list for the gradients
    gradients = []

    # Zip the coordinates from the first and second image together
    coords = zip(coords_1, coords_2)

    # Iterate over each pair of coordinates
    for coord in coords:
        try:
            # Calculate the gradient between the x-coordinates divided by the difference in y-coordinates
            # This is essentially calculating the slope between two points
            gradients.append((coord[1][0] - coord[0][0]) / (coord[1][1] - coord[0][1]))
        except:
            # If there is a division by zero error (when the difference in y-coordinates is zero), ignore it and continue to the next pair
            pass

    # Return the list of gradients
    return gradients


def filter_matching_coordinates(coords_1, coords_2):
    """Filters the matching coordinates using the gradients between them."""
    
    # Initialize empty lists for the filtered coordinates of the matched keypoints in the first and second image, and their indices
    filtered_coords_1 = []
    filtered_coords_2 = []
    filtered_coords_indices = []

    # Calculate the gradients between the coordinates of the matched keypoints in the first and second image
    gradients = calculate_gradients(coords_1, coords_2)

    # Filter the gradients using a Gaussian kernel density estimation and argrelextrema to find the maxima of the KDE
    filtered_gradients = filter_gradients(gradients)

    # Iterate over each gradient
    for i in range(len(gradients)):
        # If the gradient is in the list of filtered gradients, add the corresponding coordinates and index to the filtered lists
        if gradients[i] in filtered_gradients:
            filtered_coords_1.append(coords_1[i])
            filtered_coords_2.append(coords_2[i])
            filtered_coords_indices.append(i)

    # Return the filtered coordinates and their indices
    return filtered_coords_1, filtered_coords_2, filtered_coords_indices


def find_matching_coordinates(keypoints_1, keypoints_2, matches: list[Any]):
    """Finds the matching coordinates between the two images using the matches."""
    
    # Initialize empty lists for the coordinates of the matched keypoints in the first and second image
    coordinates_1 = []
    coordinates_2 = []

    # Iterate over each match
    for match in matches:
        # Get the index of the matched keypoint in the first image
        image_1_idx = match.queryIdx
        # Get the index of the matched keypoint in the second image
        image_2_idx = match.trainIdx
        # Get the coordinates of the matched keypoint in the first image
        (x1, y1) = keypoints_1[image_1_idx].pt
        # Get the coordinates of the matched keypoint in the second image
        (x2, y2) = keypoints_2[image_2_idx].pt
        # Add the coordinates to the respective lists
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))

    # Filter the matching coordinates using the gradients between them
    filtered_coords_1, filtered_coords_2, filtered_coords_indices = (
        filter_matching_coordinates(coordinates_1, coordinates_2)
    )

    # Return the filtered coordinates and their indices
    return filtered_coords_1, filtered_coords_2, filtered_coords_indices


def calculate_matches(descriptors_1, descriptors_2) -> list[Any]:
    """Calculates the matches between the descriptors of the two images using the brute force matcher."""
    
    # Create a Brute-Force matcher object with Hamming distance as a measure and crossCheck enabled
    brute_force = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Use the Brute-Force matcher to find matches between the descriptors of the first and second image
    matches = brute_force.match(descriptors_1, descriptors_2)

    # Sort the matches based on their distance, which represents the similarity between the matched descriptors
    matches = sorted(matches, key=lambda x: x.distance)

    # If there are no matches, return an empty list and False
    if not matches:
        return [], False

    # If there are matches, return the list of matches and True
    return matches, True


def calculate_features(image_1: Any, image_2: Any, feature_number: int):
    """Calculates the features of the two images using the ORB algorithm."""
    
    # Create an ORB (Oriented FAST and Rotated BRIEF) detector with the specified number of features
    orb = cv.ORB_create(nfeatures=feature_number)

    # Detect ORB features and compute descriptors for the first image
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)

    # Detect ORB features and compute descriptors for the second image
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)

    # Return the keypoints and descriptors for both images
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def convert_to_cv(image_1: str, image_2: str) -> Tuple[Any, Any]:
    """Converts the images to an opencv format"""

    # Read the first image from the file and convert it to grayscale using OpenCV
    image_1_cv = cv.imread(image_1, cv.IMREAD_GRAYSCALE)

    # Read the second image from the file and convert it to grayscale using OpenCV
    image_2_cv = cv.imread(image_2, cv.IMREAD_GRAYSCALE)

    # Return the grayscale images
    return image_1_cv, image_2_cv


def process_image(speeds: list[float], images: Queue) -> Tuple[list[int], Queue]:
    """Processes the images and turns the images into a speed travelled between them."""

    # Check if there are at least two images in the queue
    if images.qsize() <= 1:
        return

    # Get the first two images from the queue and put the second one back
    image1: str = images.get()
    image2: str = images.get()
    images.put(image2)

    # Convert the images to OpenCV format
    image1_cv, image2_cv = convert_to_cv(image1, image2)

    # Calculate the features of the images using the ORB algorithm
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(
        image1_cv, image2_cv, 1000
    )

    # Calculate the matches between the descriptors of the two images
    matches, success = calculate_matches(descriptors_1, descriptors_2)

    # If no matches were found, return the current speeds and images
    if not success:
        return speeds, images

    # Find the matching coordinates between the two images
    coordinates_1, coordinates_2, matches_indices = find_matching_coordinates(
        keypoints_1, keypoints_2, matches
    )

    # Filter the matches based on their indices
    matches = [matches[i] for i in matches_indices]

    # Calculate the mean distance between the coordinates of the two images
    average_feature_distance: list[float] = calculate_mean_distance(
        coordinates_1, coordinates_2
    )

    # Get the ground sample distance of the ISS
    GSD: int = get_GSD(ISS)

    # Calculate the speed of the ISS in kilometers per second
    speed: float = calculate_speed_in_kmps(average_feature_distance, GSD)

    # Add the calculated speed to the list of speeds
    speeds.append(speed)

    # Return the updated list of speeds and the queue of images
    return speeds, images


def capture(image_counter: int, speeds: list[int], images: Queue) -> int:
    """Captures an image and adds it to the queue of images to be processed"""

    # Construct the image file name using the image counter
    image_name: str = os.path.join(photos_dir, f"image{image_counter}.jpg")

    # Capture an image using the camera and save it with the constructed file name
    cam.capture(image_name)

    # Add the image file name to the images queue
    images.put(image_name)

    # If there are more than one images in the queue, process the images to calculate the speed
    # Otherwise, keep the speeds list as it is
    speeds, images = (
        process_image(speeds, images) if images.qsize() > 1 else speeds,
        images,
    )

    # If the image counter is more than 1, remove the second last image file
    if image_counter > 1:
        os.remove(os.path.join(photos_dir, f"image{image_counter-2}.jpg"))

    # Return the incremented image counter
    return image_counter + 1


def main():
    # Initialize an empty list to store the calculated speeds
    speeds = []

    # Initialize a queue to store the images to be processed
    images = Queue()

    # Initialize a counter for the images
    image_counter = 0

    # Get the current time in seconds since the epoch
    start: float = time()

    # Keep capturing images and calculating speeds until 590 seconds have passed
    while (time() - start) < 590:
        # Capture an image, add it to the images queue, process the images to calculate the speed, and increment the image counter
        image_counter: int = capture(image_counter, speeds, images)

    # Open a file named "results.txt" in write mode
    with open("results.txt", "w") as results_txt:
        # Write the mean of the calculated speeds to the file, formatted to 5 decimal places
        results_txt.write(f"{np.mean(speeds):.5f}")
    
    os.remove(os.path.join(photos_dir, f"image{image_counter-2}.jpg"))
    os.remove(os.path.join(photos_dir, f"image{image_counter-1}.jpg"))


if __name__ == "__main__":
    main()
