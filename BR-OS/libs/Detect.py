from math import pi
from re import S
import cv2
import math
from cv2 import MARKER_CROSS
from nbformat import write
import numpy as np
import Util
import json


class Well_Detector():

    write_images = False
    debug_show = False

    # default settings!

    DPI = 300  # PPI
    CRAD = 38
    big_R = 272

    HG_DP = 1.6  # 1/px inverse res, higher = better
    HG_MIN_DIST = 80  # px
    HG_MAX_RAD = 45  # px
    HG_MIN_RAD = 30

    N_wells = 12

    wells = []

    debug_ims = []

    def __init__(self, debugging=False, write_ims=False):
        self.debug_show = debugging
        self.write_images = write_ims

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
        crc = self.get_box_and_circles(im_name)
        output = cv2.imread(im_name)

        for (x, y, r) in crc:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)

        cv2.imshow(output)
        cv2.waitKey()

    def get_rect(self, im, grayim):

        temp_copy = im.copy()

        THRES_VAL = 60

        ret, thresh = cv2.threshold(grayim, THRES_VAL, 255, 0)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rects = []

        for cont in contours:

            approx = cv2.approxPolyDP(cont, 1, True)

            if len(approx) < 3:
                continue

            cv2.drawContours(temp_copy, [approx], 0, (255, 195, 14), 2)

            if cv2.arcLength(cont, True) < 1500:
                continue

            approx = cv2.approxPolyDP(approx, 100, True)
            approx = cv2.approxPolyDP(approx, 300, True)

            if len(approx) < 4:
                continue

            area = cv2.contourArea(approx)

            if area < 1e4:
                continue

            # hull = cv2.convexHull(approx)
            hull = approx

            if len(hull) == 4:
                u = list(hull.reshape(4, 2))
                xs, ys = zip(*u)
                center = (sum(xs)/4, sum(ys)/4)

                rects.append(center)

            cv2.drawContours(im, [hull], 0, (255, 195, 14), 2)
            cv2.drawMarker(im, (round(center[0]), round(
                center[1])), (255, 195, 14), markerType=cv2.MARKER_CROSS)

        # if self.debug_show:
            # self.debug_ims.extend([thresh, temp_copy, im])

        # if self.write_images:
        #     path = './Automated-scan/dev/out'
        #
        #     for i in range(4):
        #         cv2.imwrite(path + str(i) + '.bmp', ims[i])

        ret = None
        if rects:
            ret = rects[-1]
        return ret

    def return_wells(self, im_name):
        image = cv2.imread(im_name)

        # temp_im = cv2.imread(im_name)

        gray_im, gray_eq = self.process_im(image)

        if self.debug_show:
            self.debug_ims = [gray_eq, image]

        rect_cent, crc = self.get_box_and_circles(image, gray_im, gray_eq)

        if rect_cent is None:
            print('error, rect not det')
            return None

        points = self.get_circles_in_box(rect_cent, crc)

        print('viable points = '+str(len(points)))

        self.fit_wells(points)

        if self.write_images or self.debug_ims:
            # output = cv2.imread(im_name)  # output = cv2.imread(im_name)
            index = 0

            for (x, y, r) in crc:
                # for (x,y,r) in crc:
                color = (0, 10, 230)
                if (x, y) in points:
                    color = (19, 163, 235)

                cv2.circle(image, (round(x), round(y)), round(r), color, 1)

            for (x, y, r) in self.wells:
                # for (x,y,r) in crc:
                cv2.circle(image, (x, y), r, (44, 235, 140), 1)

                cv2.putText(image, str(
                    index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 195, 14), 3, cv2.LINE_AA)

                index += 1

        if self.write_images:
            new_name = im_name[:-4] + '_post' + im_name[-4:]
            # new_name2 = im_name[:-4] + '_post2' + im_name[-4:]
            cv2.imwrite(new_name, image)
            # cv2.imwrite(new_name2, temp_im)
        if self.debug_ims:
            for im in self.debug_ims:
                cv2.imshow(None, im)
                cv2.waitKey(0)

        return self.wells

    def find_offset(self,im_name, orig_center):
        image = cv2.imread(im_name)
        gray_im, _ = self.process_im(image)

        rect_center = self.get_rect(image, gray_im)

        dy = round(rect_center[1] - orig_center[1])

        if abs(dy) <= 1:
            dy = 0

        return (0, dy)

    def return_wells_2(self, im_name):
        image = cv2.imread(im_name)
        gray_im, gray_eq = self.process_im(image)

        if self.debug_show:
            self.debug_ims = [gray_eq, image]
        pass

        rect_cent, crc = self.get_box_and_circles(image, gray_im, gray_eq)

        if rect_cent is None:
            print('error, rect not det')
            return None

        points = tuple((x,y) for (x,y,r) in crc)
        
        if not points:
            print('no points detected, ERROR')
            return False

        _, _, rotation = Well_Detector.fit_polygon(
            points, 2*self.N_wells)

        self.wells = Well_Detector.recover_points(
            rect_cent, self.big_R, rotation, self.CRAD, self.N_wells)\

        if self.write_images or self.debug_ims:
            # output = cv2.imread(im_name)  # output = cv2.imread(im_name)
            index = 0

            for (x, y, r) in crc:
                # for (x,y,r) in crc:
                color = (0, 10, 230)
                # if (x, y) in points:
                #     color = (19, 163, 235)

                cv2.circle(image, (round(x), round(y)), round(r), color, 1)

            for (x, y, r) in self.wells:
                # for (x,y,r) in crc:
                cv2.circle(image, (x, y), r, (44, 235, 140), 1)

                cv2.putText(image, str(
                    index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 195, 14), 3, cv2.LINE_AA)

                index += 1

        if self.debug_ims:
            for im in self.debug_ims:
                cv2.imshow(None, im)
                cv2.waitKey(0)

        return self.wells

        

        

    def get_box_and_circles(self, im, gray_im, gray_eq_im):

        rect_center = self.get_rect(im, gray_im)  # returns center of rectangle

        circles = cv2.HoughCircles(gray_eq_im, cv2.HOUGH_GRADIENT, self.HG_DP,
                                   self.HG_MIN_DIST, minRadius=self.HG_MIN_RAD, maxRadius=self.HG_MAX_RAD)

        return rect_center, circles[0]

    def get_circles_in_box(self, rect_cent, circs):
        points = []
        for (x, y, r) in circs:
            dist = (x-rect_cent[0])**2 + (y - rect_cent[1])**2
            if Well_Detector.approx_equal(dist, self.big_R**2, 5e-2):
                points.append((x, y))
        return points

    def process_im(self, im):

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        gray = cv2.bilateralFilter(gray, 20, 20, 0)

        gray_eq = cv2.equalizeHist(gray)

        return gray, gray_eq

    def fit_wells(self, points):

        if not points:
            print('no points detected, ERROR')
            return False

        c_star, rad_star, rot_star = Well_Detector.fit_polygon(
            points, self.N_wells)

        self.wells = Well_Detector.recover_points(
            c_star, rad_star, rot_star, self.CRAD, self.N_wells)

    def recover_points(center, bigR, rot, smallR, N):
        ideal_angles = list(np.linspace(
            0, 2*pi*(1 - 1/N), N))
        return [(round(bigR*math.cos(p + rot - pi/2) + center[0]),
                 round(bigR*math.sin(p + rot - pi/2) + center[1]), smallR) for p in ideal_angles]

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

    def fit_polygon(points, N):

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
