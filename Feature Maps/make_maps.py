"""
Program to generate feature maps of distance from coastline,
distance from city, and distance from forested areas.
"""
import cv2
import numpy as np

def find_land(image):
    """
    Makes binary image indicating where is land and where isn't.
    """
    rows = np.shape(image)[0]
    cols = np.shape(image)[1]
    sea_color = image[0][0]
    is_land = np.zeros(np.shape(image)[0:2])

    for row in range(0, rows):
        for col in range(0, cols):
            is_land[row][col] = not np.array_equal(sea_color, image[row][col])

    return is_land

def find_coastline(image):
    """
    Makes binary image indicating where the coastline is.
    """
    rows = np.shape(image)[0]
    cols = np.shape(image)[1]
    sea_color = image[0][0]
    is_coast = np.zeros(np.shape(image)[0:2])

    for row in range(0, rows):
        coast_found = False
        for col in range(0, cols):
            if not np.array_equal(sea_color, image[row][col]) and not coast_found:
                is_coast[row][col] = 1
                coast_found = True

    return is_coast

def find_forest(image):
    """
    Makes binary image indicating where forest/mountain areas are.
    """
    rows = np.shape(image)[0]
    cols = np.shape(image)[1]
    forest_color = [207, 230, 193]
    is_forest = np.zeros(np.shape(image)[0:2])

    for row in range(0, rows):
        for col in range(0, cols):
            is_forest[row][col] = np.allclose(forest_color, image[row][col], rtol=0.1, atol=0.5)

    kernel = np.ones((3, 3))
    is_forest = cv2.morphologyEx(is_forest, cv2.MORPH_OPEN, kernel)
    is_forest = cv2.morphologyEx(is_forest, cv2.MORPH_CLOSE, kernel)

    return is_forest

def find_nearest_white(img, target):
    """
    Function to find distance to nearest white pixel.
    """
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:, :, 0] - target[0]) ** 2 + (nonzero[:, :, 1] - target[1]) ** 2)

    return np.ndarray.min(distances)

def find_distances(image):
    """
    Finds how "far" a point is from a white pixel.
    """
    rows = np.shape(image)[0]
    cols = np.shape(image)[1]
    distances = np.zeros(np.shape(image)[0:2])

    for row in range(0, rows):
        for col in range(0, cols):
            if image[row][col] == 0.0:
                distances[row][col] = find_nearest_white(image, (col, row))

    return distances / np.ndarray.max(distances)

def make_maps():
    """
    Create each binary image and feature map.
    """
    west = cv2.imread('maps/map_west_1.png')

    # These three groups of code create and show the binary images for each feature
    land = find_land(west)
    cv2.imshow('land', land)
    cv2.waitKey(0)

    coastline = find_coastline(west)
    cv2.imshow('coast', coastline)
    cv2.waitKey(0)

    forest = find_forest(west)
    cv2.imshow('forest', forest)
    cv2.waitKey(0)

    # These three groups will show the heatmaps made from the binary images
    
    # l_dists = find_distances(land)
    # l_map = cv2.applyColorMap(np.uint8(255 * l_dists), cv2.COLORMAP_JET)
    # cv2.imshow('land heatmap', l_map)
    # cv2.waitKey(0)

    # c_dists = find_distances(coastline)
    # c_map = cv2.applyColorMap(np.uint8(255 * c_dists), cv2.COLORMAP_JET)
    # cv2.imshow('coasr heatmap', c_map)
    # cv2.waitKey(0)

    # f_dists = find_distances(forest)
    # f_map = cv2.applyColorMap(np.uint8(255 * f_dists), cv2.COLORMAP_JET)
    # cv2.imshow('forest heatmap', f_map)
    # cv2.waitKey(0)

    # These three will save the maps to a file

    # cv2.imwrite('maps/land_heatmap.png', l_map)
    # print('land done')
    # cv2.waitKey(0)

    # cv2.imwrite('maps/coast_heatmap.png', c_map)
    # print('coast done')
    # cv2.waitKey(0)

    # cv2.imwrite('maps/forest_heatmap.png', f_map)
    # print('forest done')
    # cv2.waitKey(0)

if __name__ == '__main__':
    make_maps()
