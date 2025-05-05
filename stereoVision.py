import cv2 as cv
import numpy as np

# used the following
# https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system
# https://albertarmea.com/post/opencv-stereo-camera/
# for learning how to implement functions

def decompose_E(E):
    U, S, Vt = np.linalg.svd(E)

    # ensure both U and Vt are positive
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    
    # define rotation matrix as in slide 36
    W = np.array([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])
    
    # construct possible positive rotation matrices
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))


    # t is in the 3rd column of U
    t1 = U[:, 2]
    t2 = U[:, 2]

    if t1[2] < 0:
        t1 *= -1
    if t2[2] < 0:
        t2 *= -1

    # pick positive depth rotation and translation matrices
    if t1[2] > t2[2]:
        return R1, t1
    else:
        return R2, t2


def calibration(img1, img2, k1):

    print("calibrating...")

    gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # a. Identify matching features between the two images
    # in each dataset using any feature matching algorithms.

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    bf = cv.BFMatcher()
    matches = bf.match(des1, des2) # use brute force for matching descriptors

    # extract keypoints that match
    img1_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    img2_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # b. Estimate the Fundamental matrix using RANSAC method based on the matched features.
    # find fundamental matix using RANSAC with a threshold of 3 and 99% confidence
    F, inliers_mask = cv.findFundamentalMat(img1_pts, img2_pts, cv.FM_RANSAC, 3.0, 0.99) # convern: mess with these params
    print("F: ", F)

    # filter out outlier points
    img1_pts = img1_pts[inliers_mask.ravel() == 1]
    img2_pts = img2_pts[inliers_mask.ravel() == 1]
    img1_pts = img1_pts.reshape((img1_pts.shape[0] * 2, 1))
    img2_pts = img2_pts.reshape((img2_pts.shape[0] * 2, 1))

    # use img2 keypoints to compute epipoles of img1 then draw
    lines1 = cv.computeCorrespondEpilines(img2_pts.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3) # makes lines in format [a b c] where ax + by + c = 0

    lines2 = cv.computeCorrespondEpilines(img1_pts.reshape(-1, 1, 2), 2, F)
    lines2 = lines2.reshape(-1, 3)

    # c. Compute the Essential matrix from the Fundamental matrix 
    # considering calibration parameters.

    K = k1 # sets of image have same k matrix

    E = np.dot(np.dot(np.transpose(K), F), K)
    print("E: ", E)

    # d. Decompose the Essential matrix into rotation and translation matrices.
    R, t = decompose_E(E)
    print("R: ", R, "\nt:", t)

    return F, E, img1_pts, img2_pts, lines1, lines2


def rectify(img1, img2, img1_pts, img2_pts, F, k1, k2, lines1, lines2):
    shape = img1.shape[:2]
    print('img1.shape: ', shape)

    # gather the homography matrices used in rectification
    _, H1, H2 = cv.stereoRectifyUncalibrated(img1_pts, img2_pts, F, (1920, 1080))
    print("H1: ", H1)
    print("H2: ", H2)

    img1_warped = cv.warpPerspective(img1, H1, img1.shape[:2][::-1])
    img2_warped= cv.warpPerspective(img2, H2, img2.shape[:2][::-1])
    cv.imshow("img1 warped w/o epilines", img1_warped)
    cv.imshow("img2 warped w/o epilines", img2_warped)

    # draw epipolar lines before warping
    for _, line in enumerate(lines1):
        # solve for x0, y0 using y = (-ax-c)/b we set x=0 and y0 = -c/b hence -line[2]/line[1]
        x0, y0 = map(int, [0, -line[2] / line[1]]) 
        # at x = x1, y1 = -(ax1-c)/b where x1 = img1.shape[1], c = line[2], a = line[0], b = line[1]
        x1, y1 = map(int, [img1.shape[1], -(line[2] + line[0] * img1.shape[1]) / line[1]])
        img1_epilines = cv.line(img1, (x0, y0), (x1, y1), (255, 255, 255), 1)

    for _, line in enumerate(lines2):
        # solve for x0, y0 using y = (-ax-c)/b we set x=0 and y0 = -c/b hence -line[2]/line[1]
        x0, y0 = map(int, [0, -line[2] / line[1]]) 
        # at x = x1, y1 = -(ax1-c)/b where x1 = img1.shape[1], c = line[2], a = line[0], b = line[1]
        x1, y1 = map(int, [img2.shape[1], -(line[2] + line[0] * img2.shape[1]) / line[1]])
        img2_epilines = cv.line(img2, (x0, y0), (x1, y1), (255, 255, 255), 1)

    img1_warped_epilines = cv.warpPerspective(img1_epilines, H1, img1_epilines.shape[:2][::-1])
    img2_warped_epilines = cv.warpPerspective(img2_epilines, H1, img2_epilines.shape[:2][::-1])


    return img1_warped, img2_warped

def disparity(img1_warped, img2_warped, focalLength, baseline):
# a. Calculate the disparity map representing the pixel-wise differences between the two images.
    stereo = cv.StereoBM.create(numDisparities=16*19, blockSize=15)
    print("computing disparity")
    img1_warped = cv.cvtColor(img1_warped, cv.COLOR_BGR2GRAY)
    img2_warped = cv.cvtColor(img2_warped, cv.COLOR_BGR2GRAY)   
    cv.imshow('img1_warped', img1_warped)
    cv.imshow('img2_warped', img2_warped)

    disparity = stereo.compute(img2_warped, img1_warped).astype(np.float32)
    cv.imshow("disparity map: ", disparity)
    # scaled_disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # cv.imshow('scaled_disparity', scaled_disparity)
    # b. Rescale the disparity map and save it as grayscale and color images using heat map conversion.
    disparity_scaled_down = disparity / 2048
    disparity_scaled = cv.convertScaleAbs(disparity_scaled_down * 255)
    cv.imshow("disparity map scaled down", disparity / 2048)

    disparity_color = cv.applyColorMap(disparity_scaled, cv.COLORMAP_JET)
    cv.imshow("disparity_color", disparity_color)

    # c. Utilize the disparity information to compute depth values for each pixel.
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid_pixels = disparity > 0
    # d. Generate a depth image representing the spatial dimensions of the scene.
    depth[valid_pixels] = (baseline * focalLength) / disparity[valid_pixels]
    print("depth matrix: ", depth)
    depth_scaled_down = depth / 2048
    depth_scaled = cv.convertScaleAbs(depth_scaled_down * 255)
    cv.imshow("depth scaled", depth / 2048)

    cv.imshow("depth map", depth)

    heat_depth = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)
    cv.imshow("heat_depth", heat_depth)

    # e. Save the depth image as grayscale and color using heat map conversion for visualization

    
    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()

def depth():
    print("depth!")



def main():

    classim0 = cv.imread("problem2_dataset/classroom/im0.png")
    classim1 = cv.imread("problem2_dataset/classroom/im1.png")
    k1_class = np.array([[1746.24, 0, 14.88],
                        [0, 1746.24, 534.11],
                        [0, 0, 1]])
    k2_class = k1_class

    storage0 = cv.imread("problem2_dataset/storageroom/im0.png")
    storage1 = cv.imread("problem2_dataset/storageroom/im1.png")
    k1_storage = np.array([[1742.11, 0, 804.90],
                        [0, 1742.11, 541.22],
                        [0, 0, 1]])
    k2_storage = k1_storage

    trap0 = cv.imread("problem2_dataset/traproop/im0.png")
    trap1 = cv.imread("problem2_dataset/traproop/im1.png")
    k1_trap = np.array([[1769.02, 0, 1271.89], 
                        [0, 1769.02, 527.17],
                        [0, 0, 1]])
    k2_trap = k1_trap

    F, E, img1_pts, img2_pts, lines1, lines2 = calibration(classim0, classim1, k1_class)
    img1_warped, img2_warped = rectify(classim0, classim1, img1_pts, img2_pts, F, k1_class, k2_class, lines1, lines2)
    disparity(img1_warped, img2_warped, 1746.24, 678.37)


if __name__ == "__main__":
    main()