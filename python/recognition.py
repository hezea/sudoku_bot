import cv2
import numpy as np
from line_utils import horizontal_intercepts_polar as hicp
from line_utils import vertical_intercepts_polar as vicp
from line_utils import lines_weighted_average as lwa
from line_utils import intersection_point

def lines_horizontal_intercept(lines, limit, delta):
    '''
    Finds "horizontal" lines from array of all lines. For each "horizontal"
    line finds two points in which it intersects vertical image borders. If two
    lines intersect borders very close to each other, replace both with a
    "weighted average" line.

    Parameters:
    lines: [[[float, float]]]
        Array of lines defined by their polar coordinates. Each of the lines
        must contain distance from origin at index 0 and angle in radians at
        index 1.
    limit: float
        Coordinate of the second image border.
    delta: float
        Threshold for distance between intersection points of two lines.

    Returns:
    [(float, (int, int), (int, int))]
        Array of horizontal lines. Each line tuple contains weight followed by
        coordinates of two points.
    '''
    lines_horizontal = []
    for line in lines:
        distance = line[0][0]
        angle = line[0][1]
        similar = False
        if abs(np.cos(angle)) < 0.1:
            intercepts = hicp((distance, angle), limit)
            new_line = (1, intercepts[1], intercepts[2])
            for line_i in range(len(lines_horizontal)):
                old_line = lines_horizontal[line_i]
                average = lwa(old_line, new_line, delta)
                if average[0]:
                    average_trunc = (average[1], average[2], average[3])
                    lines_horizontal[line_i] = average_trunc
                    similar = True
                    break
            if not similar:
                lines_horizontal.append(new_line)
    lines_horizontal.sort(key = lambda x: x[1][1])
    return lines_horizontal

def lines_vertical_intercept(lines, limit, delta):
    '''
    Finds "vertical" lines from array of all lines. For each "vertical"
    line finds two points in which it intersects horizontal image borders. If two
    lines intersect borders very close to each other, replace both with a
    "weighted average" line.

    Parameters:
    lines: [[[float, float]]]
        Array of lines defined by their polar coordinates. Each of the lines
        must contain distance from origin at index 0 and angle in radians at
        index 1.
    limit: float
        Coordinate of the second image border.
    delta: float
        Threshold for distance between intersection points of two lines.

    Returns:
    [(float, (int, int), (int, int))]
        Array of vertical lines. Each line tuple contains weight followed by
        coordinates of two points.
    '''
    lines_vertical = []
    for line in lines:
        distance = line[0][0]
        angle = line[0][1]
        similar = False
        if abs(np.sin(angle)) < 0.1:
            intercepts = vicp((distance, angle), limit)
            new_line = (1, intercepts[1], intercepts[2])
            for line_i in range(len(lines_vertical)):
                old_line = lines_vertical[line_i]
                average = lwa(old_line, new_line, delta)
                if average[0]:
                    average_trunc = (average[1], average[2], average[3])
                    lines_vertical[line_i] = average_trunc
                    similar = True
                    break
            if not similar:
                lines_vertical.append(new_line)
    lines_vertical.sort(key = lambda x: x[1][0])
    return lines_vertical

def find_lines(source, display=False, **params):
    '''
    detects lines in image and returns their coordinates

    source -- str
        file path from which to load image
    display -- bool -- False
        show image on the screen with all detected lines highlited once
        finished
    **params[votes] -- int -- 150
        threshold of intersections in Hough grid to be considered a line
    **params[color] -- int(3) -- (255, 0, 0)
        color of highlited lines on image that's shown when "display" is True
    **params[delta] -- int -- image height / 18
        the minimum difference of line intercepts on both sides of the image
        at which they are still considered different lines
    '''
    image_original = cv2.imread(source)
    image_intermediate = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    canny_min, canny_max = 50, 150
    image_intermediate = cv2.Canny(image_intermediate, canny_min, canny_max)

    hough_votes = params.get('votes', 150)
    lines = cv2.HoughLines(image_intermediate, 1, np.pi / 180, hough_votes)

    image_h = image_original.shape[0]
    image_w = image_original.shape[1]
    image_lines = np.copy(image_original) * 0
    delta = params.get('delta', image_h / 18)
    lines_horizontal = lines_horizontal_intercept(lines, image_w, delta)
    lines_vertical = lines_vertical_intercept(lines, image_h, delta)
    len_vertical = len(lines_vertical)
    len_horizontal = len(lines_horizontal)
    points = [[(0, 0)] * len_vertical for _ in range(len_horizontal)]
    for line_h in range(len_horizontal):
        line_a = (lines_horizontal[line_h][1], lines_horizontal[line_h][2])
        for line_v in range(len_vertical):
            line_b = (lines_vertical[line_v][1], lines_vertical[line_v][2])
            points[line_h][line_v] = intersection_point(line_a, line_b)[1]

    if (display):
        print(len_horizontal, len_vertical)
        lines_color = params.get('color', (255, 0, 0))
        for _, point1, point2 in lines_horizontal:
            cv2.line(image_lines, point1, point2, lines_color, 5)
        for _, point1, point2 in lines_vertical:
            cv2.line(image_lines, point1, point2, lines_color, 5)
        for row in points:
            for point in row:
                cv2.circle(image_lines, point, radius=5, color=(0, 0, 255), thickness=-1)
        image_display = cv2.addWeighted(image_original, 0.8, image_lines, 1, 0)
        cv2.imshow("Detected lines", image_display)
        cv2.waitKey(0)
    
    for v in range(len_vertical - 1):
        for h in range(len_horizontal - 1):
            delta_x = int((points[v+1][h+1][0] - points[v][h][0]) * 0.1)
            delta_y = int((points[v+1][h+1][1] - points[v][h][1]) * 0.1)
            x_1, x_2 = points[v][h][0] + delta_x, points[v+1][h+1][0] - delta_x
            y_1, y_2 = points[v][h][1] + delta_y, points[v+1][h+1][1] - delta_y
            image_crop = image_original[y_1:y_2, x_1:x_2]
            cv2.imshow("Cropped images display in progress", image_crop)
            cv2.waitKey(0)

    return [lines_horizontal, lines_vertical]

find_lines('images/test.jpg', True)
