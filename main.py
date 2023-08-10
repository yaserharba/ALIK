import json
import math
import os
from copy import copy
from queue import Queue
from ClassesInfo import ClassesInfo
import cv2
import numpy as np


def bfsPixelSearch(x, y, xyzArray):
    q = Queue()
    q.put((x, y))
    vis = {(x, y)}
    cntt = 1
    while True:
        f = q.get()
        cntt += 1
        vis.add(f)
        if cntt > 1000:
            return -1, -1
        if xyzArray[f[0], f[1], 0] != math.inf and xyzArray[f[0], f[1], 0] != -math.inf:  ######### important
            return f
        if (f[0] + 1, f[1]) not in vis:
            q.put((f[0] + 1, f[1]))
        if (f[0] - 1, f[1]) not in vis:
            q.put((f[0] - 1, f[1]))
        if (f[0], f[1] + 1) not in vis:
            q.put((f[0], f[1] + 1))
        if (f[0], f[1] - 1) not in vis:
            q.put((f[0], f[1] - 1))


def bfsPixelSearch2(x, y, upperTableMask, objectMask):
    q = Queue()
    q.put((x, y))
    vis = np.full_like(upperTableMask, 0)
    while not q.empty():
        f = q.get()
        if upperTableMask[f[0], f[1]] == 0:
            continue
        objectMask[f[0], f[1]] = 255.0
        if vis[f[0] + 1, f[1]] == 0:
            vis[f[0] + 1, f[1]] = 1
            q.put((f[0] + 1, f[1]))
        if vis[f[0] - 1, f[1]] == 0:
            vis[f[0] - 1, f[1]] = 1
            q.put((f[0] - 1, f[1]))
        if vis[f[0], f[1] + 1] == 0:
            vis[f[0], f[1] + 1] = 1
            q.put((f[0], f[1] + 1))
        if vis[f[0], f[1] - 1] == 0:
            vis[f[0], f[1] - 1] = 1
            q.put((f[0], f[1] - 1))


"""
you can choose between Box or AVOMeter or ChipsBag
"""
className = "ChipsBag"
pathStr = "Images/{0}/info.json".format(className)
with open(pathStr) as json_file:
    data = json.load(json_file)
    numberOfSamples = data["numberOfSamples"]  # get the number of images to be labeled
outputImagesPathStr = "data/images/{0}/".format(className)
os.system('mkdir -p {0}'.format(outputImagesPathStr))
outputLabelsPathStr = "data/labels/{0}/".format(className)
os.system('mkdir -p {0}'.format(outputLabelsPathStr))
canceledImageNumber = 0
for fileNumber in range(numberOfSamples):
    print(fileNumber)
    with open('Images/{0}/{1}.npy'.format(className, fileNumber), 'rb') as f:
        image = np.load(f)
        xyzArray = np.load(f)
        try:
            imageCopy = copy(image)
            cv2.imshow("OrginalImage", image)
            cv2.waitKey(0)

            kernel = np.ones((5, 5), np.uint8)

            blurred_frame = cv2.GaussianBlur(image, (5, 5), 0)  # remove the noise from color image

            hsv_Image = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
            lower_orange = np.array([3, 130, 105])  # for orange points
            upper_orange = np.array([24, 255, 255])  # for orange points

            orangeMask = cv2.inRange(hsv_Image, lower_orange, upper_orange)
            orangeMask = cv2.erode(orangeMask, kernel, iterations=5)  # for erosion the white pixels
            orangeMask = cv2.dilate(orangeMask, kernel, iterations=5)  # for dilation the white pixels
            contours, hierarchy = cv2.findContours(orangeMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            circles = []
            cv2.imshow("mask of orange points ", orangeMask)
            cv2.waitKey(0)
            for cnt in contours:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                circles.append((radius, (int(x), int(y))))
            circles.sort()
            circles.reverse()
            # find the closest points with the correct depth
            xp1, yp1 = bfsPixelSearch(circles[0][1][1], circles[0][1][0], xyzArray)
            xp2, yp2 = bfsPixelSearch(circles[1][1][1], circles[1][1][0], xyzArray)
            xp3, yp3 = bfsPixelSearch(circles[2][1][1], circles[2][1][0], xyzArray)
            if xp1 == -1 or xp2 == -1 or xp3 == -1:
                canceledImageNumber += 1
                continue

            v1 = xyzArray[xp3, yp3, :] - xyzArray[xp1, yp1, :]
            v2 = xyzArray[xp2, yp2, :] - xyzArray[xp1, yp1, :]
            # the cross product is a vector normal to the plane
            cp = np.cross(v1, v2)
            cp *= -1 if cp[2] < 0 else 1
            a, b, c = cp
            abscp = np.sqrt(a * a + b * b + c * c)
            # This evaluates a * x3 + b * y3 + c * z3 which equals d
            d = np.dot(cp, xyzArray[xp3, yp3, :])
            xyzArray = xyzArray[0:1080, :, :]

            uncompletedTableMask = (d * np.ones((1080, 1920)) - a * xyzArray[:, :, 0] - b * xyzArray[:, :,
                                                                                            1] - c * xyzArray[:, :,
                                                                                                     2]) / abscp
            uncompletedTableMask = np.nan_to_num(
                255.0 * ((np.sign(uncompletedTableMask + 0.02) - np.sign(uncompletedTableMask - 0.02)) / 2))
            kernel = np.ones((3, 3), np.uint8)
            uncompletedTableMask = cv2.erode(uncompletedTableMask, kernel, iterations=6)
            uncompletedTableMask = cv2.dilate(uncompletedTableMask, kernel, iterations=13)
            uncompletedTableMask = cv2.morphologyEx(uncompletedTableMask, cv2.MORPH_CLOSE, kernel, iterations=4)
            cv2.imshow("uncompletedTableMask", uncompletedTableMask)
            cv2.waitKey(0)
            uncompletedTableMask = uncompletedTableMask.astype(np.uint8)

            cntList = []
            contours, _ = cv2.findContours(uncompletedTableMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                area = cv2.contourArea(contour)
                cntList.append((area, contour))
            cntList.sort()
            hull = cv2.convexHull(cntList[-1][1])

            completedTableMask = np.full_like(uncompletedTableMask, 0)
            cv2.fillPoly(completedTableMask, pts=[hull], color=(255))
            completedTableMask = completedTableMask.astype(np.float64)
            completedTableMask = cv2.erode(completedTableMask, kernel, iterations=4)
            cv2.imshow("completedTableMask", completedTableMask)
            cv2.waitKey(0)
            upperTableMask = (d * np.ones((1080, 1920)) - a * xyzArray[:, :, 0] - b * xyzArray[:, :, 1] - c * xyzArray[
                                                                                                              :, :,
                                                                                                              2]) / abscp
            upperTableMask = np.nan_to_num(255.0 * ((np.sign(upperTableMask - 0.02) + 1) / 2))  ############
            upperTableMask = cv2.morphologyEx(upperTableMask, cv2.MORPH_CLOSE, kernel, iterations=30)
            upperTableMask = cv2.erode(upperTableMask, kernel, iterations=8)
            cv2.imshow("upperTableMask", upperTableMask)
            cv2.waitKey(0)
            objectMask = cv2.bitwise_and(upperTableMask, completedTableMask)
            # cv2.imshow("im333", upperTableMask)
            # cv2.waitKey(0)
            kernel = np.ones((7, 7), np.uint8)
            objectMask = cv2.erode(objectMask, kernel, iterations=5)
            cv2.imshow("objectMask", objectMask)
            cv2.waitKey(0)
            for i in range(image.shape[0]):
                done = False
                for j in range(image.shape[1]):
                    if objectMask[i, j] == 255:
                        bfsPixelSearch2(i, j, upperTableMask, objectMask)
                        done = True
                        break
                if done:
                    break
            objectMask = cv2.morphologyEx(objectMask, cv2.MORPH_CLOSE, kernel, iterations=4)
            # cv2.imshow("image", imageCopy)
            cv2.waitKey(0)
            objectMask = objectMask.astype(np.uint8)
            cntArea = 0
            contours, _ = cv2.findContours(objectMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > cntArea:
                    cnt = contour
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print("fileNumber =", fileNumber, canceledImageNumber)
            image = cv2.putText(image, str(fileNumber), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 0, 200), 2, cv2.LINE_AA)
            cv2.imshow("image", image)
            # cv2.imshow("uncompletedTableMask", uncompletedTableMask)
            # cv2.imshow("completedTableMask", completedTableMask)
            # cv2.imshow("upperTableMask", upperTableMask)
            # cv2.imshow("objectMask", objectMask)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                cv2.imwrite("data/images/{0}/{0}_{1}.png".format(className, fileNumber - canceledImageNumber),
                            imageCopy)
                with open(outputLabelsPathStr + "{0}_{1}.txt".format(className, fileNumber - canceledImageNumber),
                          "w") as outfile:
                    outfile.write("{0} {1} {2} {3} {4}\n".format(
                        ClassesInfo.getClassNumber(className),
                        round((x + w / 2) / image.shape[1], 6), round((y + h / 2) / image.shape[0], 6),
                        round(w / image.shape[1], 6), round(h / image.shape[0], 6)
                    ))
                with open(outputLabelsPathStr + "info.txt", "w") as outfile:
                    outfile.write("{}\n".format(fileNumber - canceledImageNumber))
            else:
                canceledImageNumber += 1
        except:
            canceledImageNumber += 1
            print("error in fileNumber = ", fileNumber, canceledImageNumber)
cv2.destroyAllWindows()
