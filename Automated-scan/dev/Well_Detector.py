
from math import pi
import cv2
import math
import numpy as np
# from PIL import Image
# from PIL import ImageOps


class Well_Finder():

    # gray = cv2.GaussianBlur(gray, (5,5), 5)

    # out = cv2.Sobel(gray, 6, 1, 1)

    # params:

    # res = 300dpi
    DPI = 300  # PPI
    CRAD = 38

    HG_DP = 1.5  # 1/px inverse res, higher = better
    HG_MIN_DIST = 80  # px
    HG_MAX_RAD = 45  # px
    HG_MIN_RAD = 30

    N_wells = 12

    wells = []

    def return_wells(self, im_name):
        crc = self.find_circles(im_name)
        self.extrapolate(crc)

        output = cv2.imread(im_name)
        for (x, y, r) in self.wells:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)

        new_name = im_name[:-4] + 'post' + im_name[-4:]
        cv2.imwrite(new_name, output)

        return self.wells

    def find_circles(self, im_name):

        image = cv2.imread(im_name)

        # output = image.copy()ty
        gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.HG_DP,
                                   self.HG_MIN_DIST, minRadius=self.HG_MIN_RAD, maxRadius=self.HG_MAX_RAD)

        ret = []
        # for (x, y, r) in circles:
        #     ret.append((round(x), round(y), r))
        # return ret

        return [(round(x), round(y), r) for (x, y, r) in circles[0, :]]

    def extrapolate(self, circles):
        points = [(x, y) for (x, y, r) in circles]

        centre, rad, rot = Well_Finder.fit_circle(points, self.N_wells)

        ideal_angles = list(np.linspace(
            0, 2*pi*(1 - 1/self.N_wells), self.N_wells))

        self.wells = [(round(rad*math.cos(p + rot) + centre[0]),
                       round(rad*math.sin(p + rot) + centre[1]), self.CRAD) for p in ideal_angles]

    def fit_circle(points, N):

        # min norm circle fit
        x_list = [x for (x, y) in points]
        y_list = [y for (x, y) in points]

        x_sq = [x**2 for x in x_list]
        y_sq = [y**2 for y in y_list]

        xy_pr = [x*y for (x, y) in points]

        s_x_sq = sum(x_sq)
        s_y_sq = sum(y_sq)
        s_xy_pr = sum(xy_pr)

        A = [[s_x_sq, s_xy_pr, sum(x_list)], [s_xy_pr, s_y_sq, sum(y_list)], [
            sum(x_list), sum(y_list), len(x_list)]]

        b1 = sum([(a + b)*c for a, b, c in zip(x_sq, y_sq, x_list)])
        b2 = sum([(a + b)*c for a, b, c in zip(x_sq, y_sq, y_list)])

        b = [b1, b2, s_x_sq + s_y_sq]

        A = np.array(A)
        b = np.array(b)
        params = np.linalg.lstsq(A, b, rcond=None)

        centre = (params[0][0]/2, params[0][1]/2)
        rad = math.sqrt(4*params[0][2] + params[0][1]**2 + params[0][0]**2)/2

        angles = [math.atan2(y - centre[1], x - centre[0])
                  for (x, y) in points]
        ideal_angles = list(np.linspace(0, 2*pi*(1 - 1/N), N))

        diff_list = [x - y for x in angles for y in ideal_angles]
        rot = np.mean(diff_list) + pi/N

        return centre, rad, rot
