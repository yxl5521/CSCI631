"""
 User: Yu Liang(Jasmine)
 Email: yxl5521@rit.edu
 Date: 2021/2/15
"""

import matplotlib.pyplot as plt
import cv2
import skimage.io as io


def plot_histogram(image):
    """
    Plot the histogram of sonnet image
    Use skimage to read in RGB image

    :return: None
    """
    plt.hist(image.flatten(), bins=256, range=[0, 256])
    plt.ylim(top=3000)
    plt.show()


def modify_image(img, threshold):
    img[img > threshold] = 255
    img[img <= threshold] = 0
    window = 'sonnet'
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window)


def main():
    img = io.imread('./images/sonnet.png')
    plot_histogram(img)
    modify_image(img, 150)


if __name__ == '__main__':
    main()
