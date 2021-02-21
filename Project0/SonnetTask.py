"""
 User: Yu Liang(Jasmine)
 Email: yxl5521@rit.edu
 Date: 2021/2/15
"""

import matplotlib.pyplot as plt
import cv2
import skimage.io as io


def plot_histogram():
    """
    Plot the histogram of sonnet image
    Use skimage to read in RGB image

    :return: None
    """
    img = io.imread('./images/sonnet.png')
    plt.hist(img.flatten(), bins=256, range=[0, 256])
    plt.ylim(top=3000)
    plt.show()


def main():
    plot_histogram()


if __name__ == '__main__':
    main()
