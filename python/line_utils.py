import numpy as np

def intersection_point(line_a, line_b):
    '''
    Calculates an intersection point between two lines.
    
    Parameters:
    line_a: ((float, float), (float, float))
        Coordinates of the points of the first line. Must contain x coordinates
        at indices 0 and y coordinates at indices 1.
    line_b: ((float, float), (float, float))
        Coordinates of the points of the second line. Must contain x
        coordinates at indices 0 and y coordinates at indices 1.
    
    Returns:
    (1, (int, int))
        Status 1 and tuple, containing the coordinates of the intersection
        point if possible.
    (0, (0, 0))
        Status 0 and a (0, 0) tuple if the lines are parallel.
    '''
    x_1, x_2 = line_a[0][0], line_b[0][0]
    x_3, x_4 = line_a[1][0], line_b[1][0]
    y_1, y_2 = line_a[0][1], line_b[0][1]
    y_3, y_4 = line_a[1][1], line_b[1][1]
    denominator = (x_1 - x_3) * (y_2 - y_4) - (y_1 - y_3) * (x_2 - x_4)
    if denominator != 0:
        point_x = (x_1 * y_3 - y_1 * x_3) * (x_2 - x_4)
        point_x -= (x_1 - x_3) * (x_2 * y_4 - y_2 * x_4)
        point_x = int(point_x / denominator)
        point_y = (x_1 * y_3 - y_1 * x_3) * (y_2 - y_4)
        point_y -= (y_1 - y_3) * (x_2 * y_4 - y_2 * x_4)
        point_y = int(point_y / denominator)
        return (1, (point_x, point_y))
    return (0, (0, 0))

def intersection_point_polar(line_a, line_b):
    '''
    Calculates an intersection point between two lines.
    
    Parameters:
    line_a: (float, float)
        Polar coordinates of the first line. Must contain distance from origin
        at index 0 and angle in radians at index 1.
    line_b: (float, float)
        Polar coordinates of the second line. Must contain distance from origin
        at index 0 and angle in radians at index 1.
    
    Returns:
    (1, (int, int))
        Status 1 and tuple, containing the coordinates of the intersection
        point if possible.
    (0, (0, 0))
        Status 0 and a (0, 0) tuple if the lines are parallel.
    '''
    distance_a, distance_b = line_a[0], line_b[0]
    angle_a, angle_b = line_a[1], line_b[1]
    denominator = np.sin(angle_a - angle_b)
    if denominator != 0:
        point_x = distance_b * np.sin(angle_a) - distance_a * np.sin(angle_b)
        point_x = int(point_x / denominator)
        point_y = distance_a * np.cos(angle_b) - distance_b * np.cos(angle_a)
        point_y = int(point_y / denominator)
        return (1, (point_x, point_y))
    return (0, (0, 0))

def vertical_intercepts_polar(line, limit):
    '''
    Calculates points at which given line intersects y = 0 and y = limit.

    Parameters:
    line: (float, float)
        Polar coordinates of the line. Must contain distance from origin at
        index 0 and angle in radians at index 1.
    limit: float
        Y coordinate of the second line.
    
    Returns:
    (1, (int, int), (int, int))
        Status 1 and coordinates of two intersection points if possible.
    (0, (0, 0), (0, 0))
        Status 0 and two (0, 0) tuples if given line is parallel to the y = 0.
    '''
    point_1 = intersection_point_polar(line, (0, np.pi / 2))
    point_2 = intersection_point_polar(line, (limit, np.pi / 2))
    if point_1[0] and point_2[0]: 
        return (1, point_1[1], point_2[1])
    return (0, (0, 0), (0, 0))

def horizontal_intercepts_polar(line, limit):
    '''
    Calculates points at which given line intersects x = 0 and x = limit.

    Parameters:
    line: (float, float)
        Polar coordinates of the line. Must contain distance from origin at
        index 0 and angle in radians at index 1.
    limit: float
        X coordinate of the second line.
    
    Returns:
    (1, (int, int), (int, int))
        Status 1 and coordinates of two intersection points if possible.
    (0, (0, 0), (0, 0))
        Status 0 and two (0, 0) tuples if given line is parallel to the x = 0.
    '''
    point_1 = intersection_point_polar(line, (0, 0))
    point_2 = intersection_point_polar(line, (limit, 0))
    if point_1[0] and point_2[0]: 
        return (1, point_1[1], point_2[1])
    return (0, (0, 0), (0, 0))

def lines_weighted_average(line_a, line_b, threshold):
    '''
    Attempts to calculate a weighted average of the two lines by taking pairs
    of their points and then finding a weighted average point for each pair.

    Parameters:
    line_a: (float, (float, float), (float, float))
        Weight of the first line followed by coordinates of it's two points.
    line_b: (float, (float, float), (float, float))
        Weight of the second line followed by coordinates of it's two points.
    threshold: float
        Maximum distance between points in a pair until which finding an
        average line will be attempted.
    
    Returns:
    (1, float, (int, int), (int, int))
        Status 1, new line weight and tuple, containing the coordinates of the 
        two average points that define an average line if possible.
    (0, 0, (0, 0), (0, 0))
        Status 0, weight 0 and two (0, 0) tuples if average could not be found.
    '''
    delta_1 = (((line_a[1][0] - line_b[1][0]) ** 2) + 
               ((line_a[1][1] - line_b[1][1]) ** 2)) ** 0.5
    delta_2 = (((line_a[2][0] - line_b[2][0]) ** 2) +
               ((line_a[2][1] - line_b[2][1]) ** 2)) ** 0.5
    if delta_1 < threshold and delta_2 < threshold:
        weight = line_a[0] + line_b[0]
        x1 = (line_a[1][0] * line_a[0] + line_b[1][0] * line_b[0]) / weight
        y1 = (line_a[1][1] * line_a[0] + line_b[1][1] * line_b[0]) / weight
        x2 = (line_a[2][0] * line_a[0] + line_b[2][0] * line_b[0]) / weight
        y2 = (line_a[2][1] * line_a[0] + line_b[2][1] * line_b[0]) / weight
        return (1, weight, (int(x1), int(y1)), (int(x2), int(y2)))
    return (0, 0, (0, 0), (0, 0))
