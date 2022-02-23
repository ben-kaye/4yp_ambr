import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps

# purpose of script is to obtain the wells by computer vision from a raw image and store locations in a JSON
# as well as validate with user
# idea being you run this once per experiment, verify all wells are accurately captured then proceed with the automation

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())



# image = cv2.imread(args["image"])


# import image, convert to grayscale
# image_pre = Image.open('./Scans/scan001.png').convert('L')

# # histogram equalisation
# image_pre = ImageOps.autocontrast(image_pre)
# image_pre.save('./tmp/processed_scan.png')


image = cv2.imread("./tmp/img004.png")

output = image.copy()
gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

# gray = cv2.GaussianBlur(gray, (5,5), 5)

# out = cv2.Sobel(gray, 6, 1, 1)


cv2.imshow('t',gray)
cv2.waitKey(0)
# detect circles in the image

#params::


# res = 300dpi
DPI = 300 # PPI
CRAD = 3/25.4 # inch



HG_DP = 0.1 # 1/px inverse res, higher = better
HG_MIN_DIST = 20 # px
HG_MAX_RAD = 100 # px
HG_MIN_RAD = 1


circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, HG_DP, HG_MIN_DIST, minRadius=HG_MIN_RAD, maxRadius=HG_MAX_RAD)



# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	# show the output image
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)