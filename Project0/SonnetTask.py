"""
 User: Yu Liang(Jasmine)
 Email: yxl5521@rit.edu
 Date: 2021/2/15
"""

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
from fractions import Fraction
from scipy import ndimage


def plot_histogram(image):
    """
    Plot the histogram of sonnet image
    Use skimage to read in RGB image

    :return: None
    """
    plt.hist(image.flatten(), bins=256, range=[0, 256])
    plt.ylim(top=3000)
    plt.show()


def normalization(image):
    image = image.copy()
    max = np.amax(image)
    min = np.amin(image)
    upper = 255
    lower = 0
    for pixels in image:
        for idx in range(len(pixels)):
            pixels[idx] = (pixels[idx] - min) * Fraction((upper - lower), (max - min)) + lower
    return image


def make_window(block_size):
    kernel = np.ones((block_size, block_size))
    mid = int((block_size - 1) / 2)
    kernel[mid, mid] = 0
    return kernel


def mean_threshold(image, block_size, constant):
    img = image.copy()
    window = make_window(block_size)
    result = ndimage.generic_filter(img, np.nanmean, footprint=window, mode='constant', cval=np.NaN)
    threshold = result - constant
    return threshold


def median_threshold(image, block_size, constant):
    window = make_window(block_size)
    result = ndimage.generic_filter(image, np.nanmedian, footprint=window, mode='constant', cval=np.NaN)
    threshold = result + constant
    return threshold


def otsus_method(data, possible_thresholds):
    """
    Use Otsu's Method to get the best threshold
    that can minimize the weighted variance for clusters
    :param alpha: the alpha for regularization, if None then no regularization needed
    :param possible_thresholds: all thresholds that can be test
            the quantized data points
    :param data is all data in the data set,
           either all ages or all heights in this assignment
    :return: the best threshold we found and the minimum mixed variance found
    """
    # set the initial value to infinite
    # to make the condition always tru at the first time
    best_mixed_var = np.inf

    # init best threshold to None
    best_th = None

    # make a copy of original data to make sure it won't be modified
    data = data.copy()

    for threshold in possible_thresholds:
        # left set with all data less than or equal to the threshold
        left_set = data[data <= possible_thresholds[threshold]]
        # right set is all data that's larger than the threshold
        right_set = data[data > possible_thresholds[threshold]]

        # the weight is the fraction of the left/right set of data
        wt_left = Fraction(len(left_set), len(data))
        wt_right = Fraction(len(right_set), len(data))

        # calculate the population variance of each set
        var_left = np.var(left_set)
        var_right = np.var(right_set)

        # compute the mixed variance to check if the current is the best threshold
        mixed_var = (wt_left * var_left) + (wt_right * var_right)
        if mixed_var <= best_mixed_var:
            best_mixed_var = mixed_var
            best_th = threshold
    return best_th


def find_threshold(image):
    blocks = list(range(3, 20, 2))
    constants = list(range(5, 7))
    thresholds = {}
    for block in blocks:
        for constant in constants:
            threshold = mean_threshold(image, block, constant)
            thresholds[(block, constant, 'mean')] = threshold
            threshold = median_threshold(image, block, constant)
            thresholds[(block, constant, 'median')] = threshold
    best = otsus_method(image, thresholds)
    return best


def modify_image(img, threshold, result=None):
    img = img.copy()
    img[img > threshold] = 255
    img[img <= threshold] = 0
    plt.imshow(img)
    if result is not None:
        plt.title("Block: {0}  Constant: {1} {2}".format(result[0], result[1], result[2]))
    plt.axis("off")
    plt.gray()
    plt.show()


def main():
    img = io.imread('./images/sonnet.png')
    plot_histogram(img)
    modify_image(img, 106)
    normalized = normalization(img)
    best = find_threshold(normalized)
    if best[2] == 'median':
        threshold = median_threshold(normalized, best[0], -best[1])
        modify_image(normalized, threshold, (best[0], -best[1], 'median'))
        plot_histogram(threshold)
    else:
        threshold = mean_threshold(normalized, best[0], best[1])
        modify_image(normalized, threshold, best)
        plot_histogram(threshold)


if __name__ == '__main__':
    main()
