"""
 User: Yu Liang(Jasmine)
 Email: yxl5521@rit.edu
 Date: 2021/2/15
"""

import skimage.io as io
from skimage.color import rgb2xyz
import matplotlib.pyplot as plt

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
