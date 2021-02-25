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
    """
    Turn the image from BGR to Grayscale
    :param image: The image to convert
    :return: The converted image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_kp_des(gray_img):
    """
    Get keypoint and descriptor/feature from an grayscale image
    :return: the keypoint and descriptor
    """
    sift = cv2.SIFT_create()
    keypoint, descriptor = sift.detectAndCompute(gray_img, None)
    return keypoint, descriptor


def sift(images):
    """
    Visualize the feature detected from each image

    :param images: One image from each categories
    :return: None
    """
    for img in images:
        gray = to_gray(images[img])
        kp, des = get_kp_des(gray)
        img = cv2.drawKeypoints(gray, kp, images[img], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def get_features(images):
    """
    Get all the features from each image
    Put them into a list for training
    :param images: the training images
    :return: the features
    """
    features = np.array([])
    for img in images:
        gray = to_gray(img)
        kp, des = get_kp_des(gray)
        if not features.size:
            features = np.array(des)
        else:
            features = np.concatenate((features, des), axis=0)
    return features


def train(features):
    """
    Use KMean to do the clustering to the features
    :param features: the features of the training images
    :return: The trained model
    """
    kmeans = KMeans(n_clusters=40, random_state=0).fit(features)
    return kmeans


def plot_histogram(images, model):
    """
    Plot the histogram of bag of words model
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
        plt.figure(figsize=(18, 5))
        plt.bar(x_labels, y_labels)
        plt.title("BoW histogram of {0}".format(img))
        plt.xticks(x_labels, x_labels)
        plt.xlabel("words")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.show()


def main():
    # detect features of the main images
    imgs = {}
    for img in listdir('./images/sift'):
        imgs[img] = cv2.imread('images/sift/{0}'.format(img))
    sift(imgs)
    # get features from all images to train
    train_data = []
    for folder in listdir('./images/train'):
        for img in listdir('./images/train/{0}'.format(folder)):
            train_data.append(cv2.imread('./images/train/{0}/{1}'.format(folder, img)))
    features = get_features(train_data)
    # train the model
    model = train(features)
    # use model to predict the bag of words in each main images
    plot_histogram(imgs, model)


if __name__ == '__main__':
    main()
