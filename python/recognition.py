import cv2
import numpy as np
from keras.models import load_model
from rendering import *
from itertools import combinations

def assimilate_all(lines, width, height, threshold):
    new_lines = []
    for line1 in lines:
        similar = False
        for line2i in range(len(new_lines)):
            line2 = new_lines[line2i]
            p11, p12 = line1.main_points
            p21, p22 = line2.main_points
            ds = distance(p11, p21) + distance(p12, p22)
            dd = distance(p11, p22) + distance(p12, p21)
            if min(dd, ds) < threshold:
                new_line = weighted_average(line1, line2, width, height)
                if len(new_line.main_points) == 2:
                    new_lines[line2i] = new_line
                    similar = True
                    break
        if not similar:
            new_lines.append(line1)
    return new_lines

def linear_votes(line, candidates, width, height):
    points = []
    mindiv, minind = 999999, []
    for line1 in candidates:
        point = visible_intersection(line, line1, width, height)
        points.append(point)
    for ind in combinations(range(len(points)), 10):
        div = 0
        consider = [points[i] for i in ind]
        if None in consider:
            continue
        total_dist = distance(consider[0], consider[9])
        for i in range(1, 10):
            dist = distance(consider[i - 1], consider[i])
            div += abs((total_dist / 9) - dist) / total_dist * 9
        if div < mindiv:
            mindiv = div
            minind = ind
    return minind

def horizontal_voting_session(lines, width, height):
    candidates, voters, votes = [], [], []
    new_lines = []
    for line in lines:
        if line.inclination == 'vertical':
            candidates.append(line)
            votes.append(0)
        elif line.inclination == 'horizontal':
            voters.append(line)
    candidates.sort(key=lambda x: x.main_points[0][0])
    voters.sort(key=lambda x: x.main_points[0][1])
    for line in voters:
        results = linear_votes(line, candidates, width, height)
        for i in results:
            votes[i] += 1
    res = sorted(range(len(votes)), key = lambda sub: votes[sub])[-10:]
    for i in res:
        new_lines.append(candidates[i])
    for line in voters:
        new_lines.append(line)
    return new_lines

def vertical_voting_session(lines, width, height):
    candidates, voters, votes = [], [], []
    new_lines = []
    for line in lines:
        if line.inclination == 'horizontal':
            candidates.append(line)
            votes.append(0)
        elif line.inclination == 'vertical':
            voters.append(line)
    candidates.sort(key=lambda x: x.main_points[0][1])
    voters.sort(key=lambda x: x.main_points[0][0])
    for line in voters:
        results = linear_votes(line, candidates, width, height)
        for i in results:
            votes[i] += 1
    res = sorted(range(len(votes)), key = lambda sub: votes[sub])[-10:]
    for i in res:
        new_lines.append(candidates[i])
    for line in voters:
        new_lines.append(line)
    return new_lines

def find_sudoku(source, display=False, **params):
    image_original = cv2.imread(source)
    image_intermediate = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    canny_min, canny_max = 50, 150
    image_intermediate = cv2.Canny(image_intermediate, canny_min, canny_max)

    hough_votes = params.get('votes', 100)
    lines_raw = cv2.HoughLines(image_intermediate, 1, np.pi / 180, hough_votes)

    height = image_original.shape[0]
    width = image_original.shape[1]
    image_lines = np.copy(image_original) * 0
    lines = []
    for line in lines_raw:
        new_line = get_line(line[0][0], line[0][1], width, height)
        if len(new_line.main_points) == 2 and new_line.inclination != 'none':
            if new_line.inclination == 'vertical':
                a = visible_intersection(new_line, get_line(0, np.pi / 2, width, height), width, height)
                b = visible_intersection(new_line, get_line(height, np.pi / 2, width, height), width, height)
                if a and b:
                    lines.append(new_line)
            else:
                a = visible_intersection(new_line, get_line(0, 0, width, height), width, height)
                b = visible_intersection(new_line, get_line(width, 0, width, height), width, height)
                if a and b:
                    lines.append(new_line)
    lines = assimilate_all(lines, width, height, width / 18)
    lines = horizontal_voting_session(lines, width, height)
    lines = vertical_voting_session(lines, width, height)

    if (display):
        lines_color = params.get('color', (255, 0, 0))
        for line in lines:
            point1 = line.main_points[0]
            point2 = line.main_points[1]
            cv2.line(image_lines, point1, point2, lines_color, 5)
            #cv2.circle(image_lines, point, radius=10, color=(0, 0, 255), thickness=-1)
        image_display = cv2.addWeighted(image_original, 0.8, image_lines, 1, 0)
        cv2.imshow("Detected lines", image_display)
        cv2.waitKey(0)
    """
    model = load_model("models/mymodel.keras")
    sudoku = [[0] * (len_vertical - 1) for _ in range(len_horizontal - 1)]
    for v in range(len_vertical - 1):
        for h in range(len_horizontal - 1):
            delta_x = int((points[v+1][h+1][0] - points[v][h][0]) * 0.1)
            delta_y = int((points[v+1][h+1][1] - points[v][h][1]) * 0.1)
            x_1, x_2 = points[v][h][0] + delta_x, points[v+1][h+1][0] - delta_x
            y_1, y_2 = points[v][h][1] + delta_y, points[v+1][h+1][1] - delta_y
            image_crop = image_original[y_1:y_2, x_1:x_2]
            image_28 = cv2.resize(image_crop, (28, 28), interpolation= cv2.INTER_LINEAR)
            image_theshold = cv2.cvtColor(image_28, cv2.COLOR_BGR2GRAY)
            image_data = image_theshold.reshape(1, 28, 28, 1)
            image_data = (255 - image_data) / 255
            prediction = model.predict(image_data, verbose=0)
            if prediction.max() < 0.4:
                sudoku[v][h] = 0
            else:
                sudoku[v][h] = prediction.argmax()
    for i in sudoku:
        print(i)
    """
    return None
