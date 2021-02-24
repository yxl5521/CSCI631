"""
 User: Yu Liang(Jasmine)
 Email: yxl5521@rit.edu
 Date: 2021/2/15
"""
from os import listdir
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import numpy as np


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_kp_des(gray_img):
    """
    get keypoint and descriptor
    :return:
    """
    sift = cv2.SIFT_create()
    keypoint, descriptor = sift.detectAndCompute(gray_img, None)
    return keypoint, descriptor


def sift(images):
    features = np.array([])
    for img in images:
        gray = to_gray(images[img])
        kp, des = get_kp_des(gray)
        if not features.size:
            features = np.array(des)
        else:
            features = np.concatenate((features, des), axis=0)
        img = cv2.drawKeypoints(gray, kp, images[img], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    return features


def train(features):
    kmeans = KMeans(n_clusters=8, random_state=0).fit(features)
    return kmeans


def plot_histogram(images, model):
    """
    Plot the histogram of sonnet image
    Use skimage to read in RGB image

    :return: None
    """
    for img in images:
        gray = to_gray(images[img])
        kp, des = get_kp_des(gray)
        pred_results = model.predict(des)
        histogram = {}
        for pred in pred_results:
            if pred not in histogram:
                histogram[pred] = 0
            else:
                histogram[pred] += 1
        histogram = dict(sorted(histogram.items()))
        x_labels = list(histogram.keys())
        y_labels = list(histogram.values())
        plt.bar(x_labels, y_labels)
        plt.title("BoW histogram of {0}".format(img))
        plt.xlabel("words")
        plt.xticks(x_labels)
        plt.ylabel("frequency")
        plt.show()


def main():
    imgs = {}
    for img in listdir('./images/sift'):
        imgs[img] = cv2.imread('images/sift/{0}'.format(img))
    features = sift(imgs)
    model = train(features)
    plot_histogram(imgs, model)


if __name__ == '__main__':
    main()
