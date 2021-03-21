import numpy as np

'''
Function to calculate the overlap ratio given the ground-truth and predicted bounding boxes.
'''


def find_top_left(rect):
    """
    rect is a array in format [x, y, width, height]
    x, y is bottom left
    return top left coordinate
    """
    x = rect[0] + rect[3]
    y = rect[1] + rect[3]
    return x, y


def find_bottom_right(rect):
    """
    rect is a array in format [x, y, width, height]
    x, y is bottom left
    return bottom right coordinate
    """
    x = rect[0] + rect[2]
    y = rect[1] + rect[2]
    return x, y


def overlapScore(rects1, rects2):
    avgScore = 0
    scores = []
    for i, _ in enumerate(rects1):
        rect1 = rects1[i]
        rect2 = rects2[i]

        # find out left, right, top, bottom
        # implement your code here:

        top = np.maximum(
            find_top_left(rect1)[1],
            find_top_left(rect2)[1]
        )
        bottom = np.minimum(
            find_bottom_right(rect1)[1],
            find_bottom_right(rect2)[1]
        )
        right = np.minimum(
            find_bottom_right(rect1)[0],
            find_bottom_right(rect2)[0]
        )
        left = np.maximum(
            find_top_left(rect1)[0],
            find_top_left(rect2)[0]
        )

        # area of intersection
        i = np.max((0, right - left)) * np.max((0, top - bottom))

        # combined area of two rectangles
        # x, y(bottom left point), width, height
        u = rect1[2] * rect1[3] + rect2[2] * rect2[3] - i

        # return the overlap ratio
        # value is always between 0 and 1
        score = np.clip(i / u, 0, 1)
        avgScore += score
        scores.append(score)

    return avgScore, scores
