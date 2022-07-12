## Script written to test the gradient calculation of an image, the same method was then applied to the 3D volumetric data used for the project. 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

image = Image.open('ImageForGradient/puppy.jpeg')

# Ref: https://www.delftstack.com/howto/python/convert-image-to-grayscale-python/
imgGray = image.convert('L')
# Alternative: https://www.delftstack.com/howto/python/convert-image-to-grayscale-python/

## Alternative to convert image into greyscale
# Convert image to numpy array
# image_array = np.asarray(image) # has z=3 coz it has a direction for channel (rgb)

# Ref: https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
# Alternative: https://scikit-image.org/skimage-tutorials/lectures/00_images_are_arrays.html

# r = image_array[:,:,0]
# g = image_array[:,:,1]
# b = image_array[:,:,2]
# gray = 0.3333 * r + 0.3333 * g + 0.3333 * b 

gx, gy = np.gradient(imgGray)  

# Gradient along the x-axis
plt.imshow(gx)
plt.show()

# Gradient along the y-axis
plt.imshow(gy)
plt.show()

