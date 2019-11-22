import math


def return_pixel(lat, lon):
    """Takes in a latitude and longitude coord and returns a pixel location"""

    assert(isinstance(lat, float))
    assert(isinstance(lon, float))

    map_width = 1928
    map_height = 1378

    map_lon_left = -136.18
    map_lon_right = -51.55
    map_lon_delta = map_lon_right - map_lon_left

    map_lat_bottom = 6.72
    map_lat_bottom_degree = map_lat_bottom * math.pi / 180

    x = (lon - map_lon_left) * (map_width / map_lon_delta)

    lat = lat * math.pi / 180
    world_map_width = ((map_width / map_lon_delta) * 360) / (2 * math.pi)
    map_offset_y = (world_map_width / 2 * math.log(
        (1 + math.sin(map_lat_bottom_degree))
        / (1 - math.sin(map_lat_bottom_degree))))
    y = map_height - ((world_map_width / 2 * math.log(
        (1 + math.sin(lat))
        / (1 - math.sin(lat)))) - map_offset_y)

    try:
        return round(x), round(y)
    except ValueError as v:
        print(lat, lon, x, y)
        raise v


def return_lat_lon(x, y):
    """Takes a pixel coordinate and returns a pair of geographic coords"""

    map_width = 1928
    map_height = 1378

    map_lon_left = -136.18
    map_lon_right = -51.55
    map_lon_delta = map_lon_right - map_lon_left

    map_lat_bottom = 6.72
    map_lat_bottom_degree = map_lat_bottom * math.pi / 180
    world_map_width = ((map_width / map_lon_delta) * 360) / (2 * math.pi)

    # lon is percent of width times lon delta plus offset
    lon = ((x / map_width) * map_lon_delta) + map_lon_left

    map_offset_y = (world_map_width / 2 * math.log(
        (1 + math.sin(map_lat_bottom_degree))
        / (1 - math.sin(map_lat_bottom_degree))))

    lat = math.asin(-2 / (math.exp(2 * (map_height - y + map_offset_y)
                                   / world_map_width) + 1) + 1)

    # Convert back to degreees
    lat = lat * 180 / math.pi

    return lat, lon


if __name__ == "__main__":
    width = 1928
    height = 1378

    min_lon_delta = float("inf")
    min_lat_delta = float("inf")
    start_lon = return_lat_lon(0, 0)[1]
    for x in range(1, width):
        diff = return_lat_lon(x, 0)[1] - start_lon
        if math.fabs(diff) < min_lon_delta:
            min_lon_delta = math.fabs(diff)
        start_lon += diff

    start_lat = return_lat_lon(0, 0)[0]
    for y in range(1, height):
        diff = return_lat_lon(0, y)[0] - start_lat
        if math.fabs(diff) < min_lat_delta:
            min_lat_delta = math.fabs(diff)
        start_lat += diff
    print(min_lat_delta, min_lon_delta)

    print(return_lat_lon(0, 0), return_lat_lon(width, height))
