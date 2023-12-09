import numpy as np

class ImageLine():
    """
    RAW INITIALIZATION IS VERY UNSTABLE - USE get_line() INSTEAD!
    """
    def __init__(self, rho, theta):
        self.rho = rho
        self.theta = theta
        self.weight = 1
        self.main_points = []
        self.inclination = 'none'
        if abs(np.sin(theta)) <= 0.05:
            self.inclination = 'vertical'
        elif abs(np.cos(theta)) <= 0.05:
            self.inclination = 'horizontal'

def distance(point1, point2):
    width = point1[0] - point2[0]
    height = point1[1] - point2[1]
    return ((width ** 2) + (height ** 2)) ** 0.5

def get_line(rho, theta, width, height):
    new_line = ImageLine(rho, theta)
    main_points = border_intersection(new_line, width, height)
    new_line.main_points = main_points
    return new_line

def visible_intersection(line1, line2, width, height):
    denominator = np.sin(line1.theta - line2.theta)
    if denominator != 0:
        x = line2.rho * np.sin(line1.theta) - line1.rho * np.sin(line2.theta)
        x = int(x / denominator)
        y = line1.rho * np.cos(line2.theta) - line2.rho * np.cos(line1.theta)
        y = int(y / denominator)
        if x >=0 and y >= 0 and x <= width and y <= height:
            return (x, y)
    return None

def border_intersection(line, width, height):
    borders = [(0, np.pi / 2), (height, np.pi / 2), (0, 0), (width, 0)]
    intersections = []
    for rho, theta in borders:
        border = ImageLine(rho, theta)
        new_point = visible_intersection(line, border, width, height)
        if new_point:
            intersections.append(new_point)
    return intersections

def point_weighted_average(point1, point2, weight1, weight2):
    x1, y1 = point1
    x2, y2 = point2
    weight = weight1 + weight2
    x = int((x1 * weight1 + x2 * weight2) / weight)
    y = int((y1 * weight1 + y2 * weight2) / weight)
    return (x, y)

def weighted_average(line1, line2, width, height):
    w1, w2 = line1.weight, line2.weight
    p11, p12 = line1.main_points
    p21, p22 = line2.main_points
    ds = distance(p11, p21) + distance(p12, p22)
    dd = distance(p11, p22) + distance(p12, p21)
    if dd < ds:
        p22, p21 = line2.main_points
    pa = point_weighted_average(p11, p21, w1, w2)
    pb = point_weighted_average(p12, p22, w1, w2)
    weight = w1 + w2
    theta = np.pi / 2 - np.arcsin((pa[1] - pb[1]) / distance(pa, pb))
    rho = pa[0] * np.cos(theta) + pa[1] * np.sin(theta)
    new_line = get_line(rho, theta, width, height)
    new_line.weight = weight
    return new_line
