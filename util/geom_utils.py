import math
from shapely.geometry import Polygon


def poly_iou(poly1: Polygon, poly2: Polygon):
    try:
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        return intersection_area / union_area
    except Exception as e:
        poly1 = poly1.buffer(1)
        poly2 = poly2.buffer(1)
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        return intersection_area / union_area



def is_clockwise_or_not(points):
    s = 0
    for i in range(0, len(points) - 1):
        s += points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0]
    return s > 0

def x_axis_angle(y):
    # 以图像坐标系为准，(1,0)方向记为0度，逆时针绕一圈到360度
    # print('-------------')
    # print(y)
    y_right_hand = (y[0], -y[1])
    # print(y_right_hand)

    x = (1, 0)
    inner = x[0] * y_right_hand[0] + x[1] * y_right_hand[1]
    # print(inner)
    y_norm2 = (y_right_hand[0] ** 2 + y_right_hand[1] ** 2) ** 0.5
    # print(y_norm2)
    cosxy = inner / y_norm2
    # print(cosxy)
    angle = math.acos(cosxy)
    # print(angle, math.degrees(angle))
    # print('-------------')
    return math.degrees(angle) if y_right_hand[1] >= 0 else 360 - math.degrees(angle)

def get_quadrant(angle):
    if angle[0] < angle[1]:
        if 0 <= angle[0] < 90 and 0 <= angle[1] < 90:
            quadrant = (angle[1] - angle[0], 0, 0, 0)
        elif 0 <= angle[0] < 90 and 90 <= angle[1] < 180:
            quadrant = (90 - angle[0], angle[1] - 90, 0, 0)
        elif 0 <= angle[0] < 90 and 180 <= angle[1] < 270:
            quadrant = (90 - angle[0], 90, angle[1] - 180, 0)
        elif 0 <= angle[0] < 90 and 270 <= angle[1] < 360:
            quadrant = (90 - angle[0], 90, 90, angle[1] - 270)
        elif 90 <= angle[0] < 180 and 90 <= angle[1] < 180:
            quadrant = (0, angle[1] - angle[0], 0, 0)
        elif 90 <= angle[0] < 180 and 180 <= angle[1] < 270:
            quadrant = (0, 180 - angle[0], angle[1] - 180, 0)
        elif 90 <= angle[0] < 180 and 270 <= angle[1] < 360:
            quadrant = (0, 180 - angle[0], 90, angle[1] - 270)
        elif 180 <= angle[0] < 270 and 180 <= angle[1] < 270:
            quadrant = (0, 0, angle[1] - angle[0], 0)
        elif 180 <= angle[0] < 270 and 270 <= angle[1] < 360:
            quadrant = (0, 0, 270 - angle[0], angle[1] - 270)
        elif 270 <= angle[0] < 360 and 270 <= angle[1] < 360:
            quadrant = (0, 0, 0, angle[1] - angle[0])
    else:
        if 0 <= angle[1] < 90 and 0 <= angle[0] < 90:
            quadrant_ = (angle[0] - angle[1], 0, 0, 0)
        elif 0 <= angle[1] < 90 and 90 <= angle[0] < 180:
            quadrant_ = (90 - angle[1], angle[0] - 90, 0, 0)
        elif 0 <= angle[1] < 90 and 180 <= angle[0] < 270:
            quadrant_ = (90 - angle[1], 90, angle[0] - 180, 0)
        elif 0 <= angle[1] < 90 and 270 <= angle[0] < 360:
            quadrant_ = (90 - angle[1], 90, 90, angle[0] - 270)
        elif 90 <= angle[1] < 180 and 90 <= angle[0] < 180:
            quadrant_ = (0, angle[0] - angle[1], 0, 0)
        elif 90 <= angle[1] < 180 and 180 <= angle[0] < 270:
            quadrant_ = (0, 180 - angle[1], angle[0] - 180, 0)
        elif 90 <= angle[1] < 180 and 270 <= angle[0] < 360:
            quadrant_ = (0, 180 - angle[1], 90, angle[0] - 270)
        elif 180 <= angle[1] < 270 and 180 <= angle[0] < 270:
            quadrant_ = (0, 0, angle[0] - angle[1], 0)
        elif 180 <= angle[1] < 270 and 270 <= angle[0] < 360:
            quadrant_ = (0, 0, 270 - angle[1], angle[0] - 270)
        elif 270 <= angle[1] < 360 and 270 <= angle[0] < 360:
            quadrant_ = (0, 0, 0, angle[0] - angle[1])
        quadrant = (90 - quadrant_[0], 90 - quadrant_[1], 90 - quadrant_[2], 90 - quadrant_[3])
    return quadrant

def find_which_angle_to_counterclockwise_rotate_from(t):
    if t > 270:
        return 630 - t
    else:
        return 270 - t

def counter_degree(d):
    if d >= 180:
        return d - 180
    else:
        return d + 180

def rotate_degree_clockwise_from_counter_degree(src_degree, dest_degree):
    delta = src_degree - dest_degree
    return delta if delta >= 0 else 360 + delta

def rotate_degree_counterclockwise_from_counter_degree(src_degree, dest_degree):
    delta = dest_degree - src_degree
    return delta if delta >= 0 else 360 + delta

def poly_area(points):
    s = 0
    points_count = len(points)
    for i in range(points_count):
        point = points[i]
        point2 = points[(i + 1) % points_count]
        s += (point[0] - point2[0]) * (point[1] + point2[1])
    return s / 2