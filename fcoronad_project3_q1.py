import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
# from google.colab.patches import cv2_imshow
import os

def calibrate(image_list):

    chessboard_dims = (8, 6)

    calibration_results = {}
    pts_3d = []

    # intialize world array with zeros of the shape 12x3
    pts_3d = np.zeros((chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
    # create grid of coordinates the size of the chessboard then transpose and reshape into array
    pts_3d[:, :2] = np.mgrid[0:chessboard_dims[0], 0:chessboard_dims[1]].T.reshape(-1, 2)

    points_3d = []
    points_2d = []

    for img in img_list:
        project3_path = '/home/fabrizzio/Downloads/Grad_School/673/project3/'
        print('current img is: ', img)
        pic = cv.imread(project3_path + img)
        pic = cv.resize(pic, (1920, 1080))
        gray_pic = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray_pic, chessboard_dims, None)

        if corners is not None:
            # print("corners is not None for", img)

            # define the termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # refine corner locations
            corners = cv.cornerSubPix(gray_pic, corners, (9, 9), (-1, -1), criteria)

            points_2d.append(corners) # append 2d points
            points_3d.append(pts_3d)
            cv.drawChessboardCorners(pic, chessboard_dims, corners, True)
            cv.imshow("corners on " + img + " before calibration", pic)
            # cv.waitKey(2000)
            cv.destroyAllWindows()

        else:
            print('corners is None for', img)

    print("gathering calibration coefficients...")
    # dist_coeffs is the distortion coefficient, R is rotation, T is translation
    # K is the calibration matrix
    ret, K, dist_coeffs, R, T = cv.calibrateCamera(
        points_3d, points_2d, gray_pic.shape[::-1], None, None)

    return K, dist_coeffs, R, T, points_3d, points_2d

calibration_pics = "calibrationPics/" # define folder with pictures
pictures = os.listdir(calibration_pics) # get list of all files in folder
img_list = [] # declare img_list as a list
chessboard_size = 26 # mm


# gather images in list to go through them when performing calibration
for file in pictures:
  if file.lower().endswith('.jpg'):
    img_path = os.path.join(calibration_pics, file) # identify path to image
    img_list.append(img_path) # append to list
    # print(img_path)
# print("img_list: ", img_list)

K, dist_coeffs, R, T, points_3d, points_2d = calibrate(img_list)
# print('R: ', R, "\nis of type: ", type(R))

# 3: Reprojection Error Analysis
error_reprojection = []
for i in range(len(points_3d)):
    pts_2d, _ = cv.projectPoints(points_3d[i], R[i], T[i], K, dist_coeffs)
    error_reprojection.append(cv.norm(points_2d[i], pts_2d, cv.NORM_L2) / len(pts_2d))

print("avg projection error: ", sum(error_reprojection)/len(error_reprojection))

img_nums = []
for name in img_list:
    # extract integers from img_list
    img_nums.append(int(name[20:24]))

plt.scatter(img_nums, error_reprojection, color = 'red', marker='x')
plt.xlabel('image number')
plt.ylabel('reprojection error')
plt.title('reprojection error analysis')
plt.show()

k = 0
# 4: Drawing reprojected points based off camera paramters R, T and K
for img_name in img_list:
    project3_path = '/home/fabrizzio/Downloads/Grad_School/673/project3/'
    img = cv.resize(cv.imread(project3_path + img_name), (1920, 1080)) # get img and resize
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray_img, (8, 6), None) # find corners again

    if ret:
        # refine corners using criteria and already gathered corners
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray_img, corners, (9, 9), (-1, -1), criteria)
        for corner in corners: # plot these uncalibrated corners
            x, y = corner[0]
            cv.circle(img, (int(x), int(y)), 4, (255, 0, 0), -1)            

        # convert into array so we can use in projectPoints
        corners = np.array(corners, dtype=np.float32)
        # projectPoints needs this shape
        corners = corners.reshape(-1, 1, 2)
        print("k: ", k)
        print("type(corners[k]): ", type(corners))
        print("type(R[k]): ", type(R[k]))
        print("type(T[k]): ", type(T[k]))
        print("type(K): ", type(K))
        print("points_3d[k]): ", type(points_3d[k]))

        # get projected points using camera parameters
        reprojected_pts, _ = cv.projectPoints(points_3d[k], R[k], T[k], K, dist_coeffs)

        for pt in reprojected_pts:
            i, j = pt[0]
            cv.circle(img, (int(i), int(j)), 4, (0, 0, 255), -1)
        cv.imshow('reprojected points and original points', img)
        cv.waitKey(2000)
    k += 1 
cv.destroyAllWindows()