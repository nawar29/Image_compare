# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import cv2
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB,multichannel=True)

	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()

# construct the argument parser and parse the arguments
# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}
original = cv2.imread("expertise.png")
contrast = cv2.imread("expertise - Copy.png")
compare_images(contrast, original, "Original vs. Contrast")

#



# loop over the image paths
#for imagePath in glob.glob(args["dataset"] + "/*.png"):
for file in os.listdir():
    if file.endswith('.png'):
    	# extract the image filename (assumed to be unique) and
    	# load the image, updating the images dictionary
    	#filename = imagePath[imagePath.rfind("/") + 1:]
    	image = cv2.imread(file)
    	images[file] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    	# extract a 3D RGB color histogram from the image,
    	# using 8 bins per channel, normalize, and update
    	# the index
    	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
    		[0, 256, 0, 256, 0, 256])
    	hist = cv2.normalize(hist, hist).flatten()
    	index[file] = hist