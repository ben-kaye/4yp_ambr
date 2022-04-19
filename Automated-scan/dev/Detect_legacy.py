from http.client import CONTINUE
from math import pi
from random import sample
import cv2
import math
from matplotlib.pyplot import gray
import numpy as np
import Util
# from PIL import Image
# from PIL import ImageOps

import itertools

import json


class Well_Detector():

    # default settings!

    DPI = 300  # PPI
    CRAD = 38
    big_R = 272

    HG_DP = 1.5  # 1/px inverse res, higher = better
    HG_MIN_DIST = 80  # px
    HG_MAX_RAD = 45  # px
    HG_MIN_RAD = 30

    N_wells = 12

    wells = []



    def __init__(self):
        settings = Util.load_settings()

        if settings:
            self.DPI = settings["scan_dpi"]
            self.HG_DP = settings["hg_dp"]
            self.HG_MAX_RAD = settings["hg_max_rad"]
            self.HG_MIN_RAD = settings["hg_min_rad"]
            self.HG_MIN_DIST = settings["hg_min_dist"]
            self.N_wells = settings["wells_per_cluster"]
            self.CRAD = settings["crad"]

    def show_circs(self, im_name):
        crc = self.find_circles(im_name)
        output = cv2.imread(im_name)
        
        for (x, y, r) in crc:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)

        cv2.imshow(output)
        cv2.waitKey()

    def get_rect(self,im, grayim):

        cv2.imshow(None,grayim)
        cv2.waitKey(0)

        ret, thresh = cv2.threshold(grayim,60,255,0)


        cv2.imshow(None,thresh)
        cv2.waitKey(0)


        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rects = []

        for cont in contours:
            
            approx = cv2.approxPolyDP(cont, 50, True)

            if len(approx) < 4:
                continue
            
            area = cv2.contourArea(approx)

            if area < 1e5:
                continue

            hull = cv2.convexHull(approx)

            if len(hull) == 4:
                rects.append(cv2.minAreaRect(hull))

            cv2.drawContours(im, [hull], 0, (255, 0, 0), 3)

        
        cv2.imshow(None,im)
        cv2.waitKey(0)

        return rects[-1]


    def return_wells(self, im_name):
        rect, crc = self.find_circles(im_name)
        points = self.get_circles_in_box(rect,crc)

        self.fit_wells(points)

        output = cv2.imread(im_name)

        index = 0
        for (x, y, r) in self.wells:
            # for (x,y,r) in crc:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            
            output = cv2.putText(output, str(
                index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
                
            index += 1

        new_name = im_name[:-4] + '_post' + im_name[-4:]
        cv2.imwrite(new_name, output)

        return self.wells

    def find_circles(self, im_name):

        image = cv2.imread(im_name)

        gray_im = self.process_im(image)

        rect = self.get_rect(image,gray_im)

        circles = cv2.HoughCircles(gray_im, cv2.HOUGH_GRADIENT, self.HG_DP,
                                   self.HG_MIN_DIST, minRadius=self.HG_MIN_RAD, maxRadius=self.HG_MAX_RAD)

        return rect, [(round(x), round(y), round(r)) for (x, y, r) in circles[0, :]]

    def get_circles_in_box(self, rect, circles):
        center = rect[0]


        points = []
        for (x,y,r) in circles:
            dist = (x-center[0])**2 + (y - center[1])**2
            if Well_Detector.approx_equal(dist,self.big_R**2,5e-2):
                points.append((x,y))
        return points




    def process_im(self, im):
        gray = cv2.equalizeHist(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

        # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        gray = cv2.bilateralFilter(gray,20,20,0)

        # cv2.bilateralFilter()

        # gray = cv2.GaussianBlur(gray, (5,5), 5)
        # gray = cv2.Sobel(gray, 6, 1, 1)
        return gray

    def fit_wells(self,points):
        c_star, rad_star, rot_star = Well_Detector.fit_circle(
            points, self.N_wells)

        self.wells = Well_Detector.recover_points(
            c_star, rad_star, rot_star, self.CRAD, self.N_wells)

    def extrapolate(self, circles):
        fit_thresh = 2
        key_interval = 10
        sample_N = 3
        rad_thresh = 10e-3

        points = [(x, y) for (x, y, r) in circles]

        c_star = 0
        rot_star = 0
        rad_star = 0

        final_points = {}

        for sample_points in itertools.combinations(points, sample_N):
            centre, rad, rot = Well_Detector.fit_circle(
                sample_points, self.N_wells)

            apr_c = (Well_Detector.discretise(
                centre[0], key_interval), Well_Detector.discretise(centre[1], key_interval))

            # does it match expected radius and are the points on a circle

            valid_set = False

            if Well_Detector.approx_equal(rad, self.big_R, rad_thresh):
                if sample_N > 3:
                    if Well_Detector.fits_circle(sample_points, centre, rad, fit_thresh):
                        valid_set = True
                else:
                    valid_set = True

            if valid_set:
                if apr_c in final_points:
                    final_points[apr_c].extend(sample_points)
                else:
                    final_points[apr_c] = list(sample_points)

        max_length = 0

        best_points = []
        for k in final_points:
            f = final_points[k]

            # if len(f) > 4:
            z = list(dict.fromkeys(f))
            if len(z) > max_length:

                c1, r1, rot1 = Well_Detector.fit_circle(
                    z, self.N_wells)

                if Well_Detector.fits_circle(z, c1, r1, fit_thresh):
                    max_length = len(z)
                    best_points = z

        c_star, rad_star, rot_star = Well_Detector.fit_circle(
            best_points, self.N_wells)

        self.wells = Well_Detector.recover_points(
            c_star, rad_star, rot_star, self.CRAD, self.N_wells)

        print('wells detected: '+str(len(best_points)))

        # self.wells = [(x, y, self.CRAD) for (x, y) in best_points]
        # self.wells = [(x, y, self.CRAD) for (x, y) in points]

    def recover_points(center, bigR, rot, smallR, N):
        ideal_angles = list(np.linspace(
            0, 2*pi*(1 - 1/N), N))
        return [(round(bigR*math.cos(p + rot - pi/2) + center[0]),
                 round(bigR*math.sin(p + rot- pi/2) + center[1]), smallR) for p in ideal_angles]

    def fits_circle(points, center, rad, thresh):
        fits = True

        for (x, y) in points:
            u = x - center[0]
            v = y - center[1]

            ang_a = 2*math.atan((math.sqrt(u**2 + v**2)-u)/v)
            ang_b = 2*math.atan((-math.sqrt(u**2 + v**2)-u)/v)
            dist2_a = (u - rad*math.cos(ang_a))**2 + \
                (v - rad*math.sin(ang_a))**2
            dist2_b = (u - rad*math.cos(ang_b))**2 + \
                (v - rad*math.sin(ang_b))**2

            if min(dist2_a, dist2_b) > thresh ** 2:
                fits = False

        return fits

    def discretise(val, interval):
        return round(val - val % interval)

    def approx_equal(val, desired_val, error):
        return val < desired_val*(1 + error) and val > desired_val*(1 - error)

    def fit_circle(points, N):

        # currently slightly bugged with the angle offset????

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

        center = (params[0][0]/2, params[0][1]/2)
        rad = math.sqrt(4*params[0][2] + params[0][1]**2 + params[0][0]**2)/2

        # angles =
        raw_angles = [(math.atan2(y - center[1], x - center[0]))
                      for (x, y) in points]

        sep = 2*pi/N

        angles = []
        for r in raw_angles:
            r_norm = r + 2*pi if r < 0 else r
            res = r_norm % sep

            if res > sep/2:
                res -= sep

            angles.append(res)

        rot = np.mean(angles)

        return center, rad, rot

    def store_well_loc(circles):
        circle_loc = {}
        circle_loc['wells'] = [{'center': [x, y], 'radius': r}
                               for (x, y, r) in circles]

        with open('./Automated-scan/data/well_locations.json', 'w') as outfile:
            json.dump(circle_loc, outfile)

    def read_well_loc():
        with open('./Automated-scan/data/well_locations.json') as json_file:
            data = json.load(json_file)
        return [(well['center'][0], well['center'][1], well['radius']) for well in data['wells']]