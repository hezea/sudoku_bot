import cv2
import numpy as np

def transform_line(distance, angle, image_size):
    '''
    TODO: add info
    '''
    if abs(np.cos(angle)) < 0.1:
        intercept_max_x = image_size[1]
        intercept_max_y = (-np.cos(angle) / np.sin(angle)) * image_size[1]
        intercept_max_y += distance / np.sin(angle)
        intercept_min = (0, int(distance / np.sin(angle)))
        intercept_max = (int(intercept_max_x), int(intercept_max_y))
        return (1, intercept_min, intercept_max)
    elif abs(np.sin(angle)) < 0.1:
        intercept_max_y = image_size[0]
        intercept_max_x = image_size[0] * np.sin(angle) - distance
        intercept_max_x /= -np.cos(angle)
        intercept_min = (int(distance / np.cos(angle)), 0)
        intercept_max = (int(intercept_max_x), int(intercept_max_y))
        return (2, intercept_min, intercept_max)
    else:
        return (0, 0, 0)

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

    #blur_kernel = (5, 5)
    #image_intermediate = cv2.GaussianBlur(image_intermediate, blur_kernel, 0)

    canny_min, canny_max = 50, 150
    image_intermediate = cv2.Canny(image_intermediate, canny_min, canny_max)

    hough_votes = params.get('votes', 150)
    lines = cv2.HoughLines(image_intermediate, 1, np.pi / 180, hough_votes)

    lines_horizontal = []
    lines_vertical = []
    delta = params.get('delta', image_original.shape[0] / 18)
    for line in lines:
        for distance, angle in line:
            new_line = transform_line(distance, angle, image_original.shape)
            if new_line[0] == 0:
                continue
            elif new_line[0] == 1:
                similar = False
                for line_i in range(len(lines_horizontal)):
                    intercept_min = lines_horizontal[line_i][0][1]
                    intercept_max = lines_horizontal[line_i][1][1]
                    weight = lines_horizontal[line_i][2]
                    delta_min = abs(intercept_min - new_line[1][1])
                    delta_max = abs(intercept_max - new_line[2][1])
                    if (delta_min < delta and delta_max < delta):
                        x1 = int((lines_horizontal[line_i][0][0] * weight + 
                                  new_line[1][0]) / (weight + 1))
                        y1 = int((lines_horizontal[line_i][0][1] * weight + 
                                  new_line[1][1]) / (weight + 1))
                        x2 = int((lines_horizontal[line_i][1][0] * weight + 
                                  new_line[2][0]) / (weight + 1))
                        y2 = int((lines_horizontal[line_i][1][1] * weight + 
                                  new_line[2][1]) / (weight + 1))
                        lines_horizontal[line_i] = [(x1, y1), (x2, y2), 
                                                    weight + 1]
                        similar = True
                        break
                if not similar:
                    lines_horizontal.append([new_line[1], new_line[2], 1])
            else:
                similar = False
                for line_i in range(len(lines_vertical)):
                    intercept_min = lines_vertical[line_i][0][0]
                    intercept_max = lines_vertical[line_i][1][0]
                    weight = lines_vertical[line_i][2]
                    delta_min = abs(intercept_min - new_line[1][0])
                    delta_max = abs(intercept_max - new_line[2][0])
                    if (delta_min < delta and delta_max < delta):
                        x1 = int((lines_vertical[line_i][0][0] * weight + 
                                  new_line[1][0]) / (weight + 1))
                        y1 = int((lines_vertical[line_i][0][1] * weight + 
                                  new_line[1][1]) / (weight + 1))
                        x2 = int((lines_vertical[line_i][1][0] * weight + 
                                  new_line[2][0]) / (weight + 1))
                        y2 = int((lines_vertical[line_i][1][1] * weight + 
                                  new_line[2][1]) / (weight + 1))
                        lines_vertical[line_i] = [(x1, y1), (x2, y2), 
                                                  weight + 1]
                        similar = True
                        break
                if not similar:
                    lines_vertical.append([new_line[1], new_line[2], 1])

    if (display):
        image_lines = np.copy(image_original) * 0
        lines_color = params.get('color', (255, 0, 0))
        for point1, point2, _ in lines_horizontal:
            cv2.line(image_lines, point1, point2, lines_color, 5)
        for point1, point2, _ in lines_vertical:
            cv2.line(image_lines, point1, point2, lines_color, 5)
        image_display = cv2.addWeighted(image_original, 0.8, image_lines, 1, 0)
        cv2.imshow("Detected lines", image_display)
        cv2.waitKey(0)
    return [lines_horizontal, lines_vertical]

find_lines('images/test.jpg', True)

