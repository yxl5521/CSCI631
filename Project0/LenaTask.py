"""
 User: Yu Liang(Jasmine)
 Email: yxl5521@rit.edu
 Date: 2021/2/15
"""

import skimage.io as io
from skimage.color import rgb2xyz
import matplotlib.pyplot as plt
import cv2


def task_1():
    """
    Convert the color space of Lena.png from RGB to CIE XYZ

    :return: None
    """
    # Read the Lena rgb image with type uint8
    rgb_img = io.imread('./images/Lena.png')
    # Convert to CIE XYZ
    xyz_img = rgb2xyz(rgb_img)
    fig, ax = plt.subplots(1, 2)
    # Display the image side by side with original
    ax[0].imshow(rgb_img)
    ax[0].set_axis_off()
    ax[0].set_title('Original(RGB) Lena')
    ax[1].imshow(xyz_img)
    ax[1].set_axis_off()
    ax[1].set_title('CIE XYZ Lena')
    plt.show()


def task_2():
    """
    Compute the color histogram of Lena.png and plot

    :return:None
    """
    # Read the Lena rgb image with type uint8
    rgb_img = io.imread('./images/Lena.png')
    # r(red) maps to 0, g(green) maps to 1, b(blue) maps to 2
    rgb_list = ['r', 'g', 'b']
    rgb_range = [0, 256]
    for idx in range(len(rgb_list)):
        color_hist = cv2.calcHist([rgb_img], [idx], None, [256], rgb_range)
        plt.plot(color_hist, color=rgb_list[idx], label=rgb_list[idx])
        plt.xlim(rgb_range)
    plt.xlabel('Channel value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def main():
    task_1()
    task_2()


if __name__ == '__main__':
    main()
