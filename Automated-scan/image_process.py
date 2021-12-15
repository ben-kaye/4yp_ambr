import cv2
import numpy as np
import json

# 64 bit process

class im_process():

    def locate_wells(scan_path):
        image = cv2.imread(scan_path)

        output = image.copy()
        gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        gray = cv2.GaussianBlur(gray, (5, 5), 5)

        out = cv2.Sobel(gray, 6, 1, 1)

        cv2.imshow('t', gray)
        cv2.waitKey(0)
        # detect circles in the image

        # params::

        HG_DP = 1000  # 1/px inverse res, higher = better
        HG_MIN_DIST = 35  # px
        HG_MAX_RAD = 15  # px
        HG_MIN_RAD = 8

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, HG_DP,
                                   HG_MIN_DIST, minRadius=HG_MIN_RAD, maxRadius=HG_MAX_RAD)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5),
                              (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            cv2.imshow("output", np.hstack([image, output]))
            cv2.waitKey(0)

        # format circles from cv2 to dict then write to JSON
        im_process.store_well_loc(circles)

    def store_well_loc(circles):
        circle_loc = {}
        circle_loc['wells'] = [{'center': [x, y], 'radius': r}
                               for (x, y, r) in circles]

        with open('./Automated-scan/well_locations.json', 'w') as outfile:
            json.dump(circle_loc, outfile)

    def read_well_loc():
        with open('./Automated-scan/well_locations.json') as json_file:
            data = json.load(json_file)
        return [(well['center'][0], well['center'][1], well['radius']) for well in data]

    def extract_well_ims(circles, image):

        well_ims = []

        for x, y, r in circles:
            well_ims.append(image[x-r:x+r, y-r:y+r])

        return well_ims
