import numpy as np

'''
Function to calculate the overlap ratio given the ground-truth and predicted bounding boxes.
'''

def overlapScore(rects1, rects2):

    avgScore = 0
    scores = []

    for i, _ in enumerate(rects1):

        rect1 = rects1[i]
        rect2 = rects2[i]

        # find out left, right, top, bottom
        # implement your code here:



        # area of intersection
        i = np.max((0, right-left))*np.max((0,bottom-top))

        # combined area of two rectangles
        u = rect1[2]*rect1[3] + rect2[2]*rect2[3] - i

        # return the overlap ratio
        # value is always between 0 and 1
        score = np.clip(i/u, 0, 1)
        avgScore += score
        scores.append(score)

    return avgScore, scores