import json
import os
from pathlib import Path

from freenect2 import Device, FrameType
import numpy as np
import cv2

# Open the default device and capture a color and depth frame.

device = Device()
frames = {}
className = "Box"

try:
    with open("data/labels/{}/info.txt".format(className)) as f:
        fileNumber = int(next(f).split()[0]) + 1
except:
    fileNumber = 0
print(fileNumber)

pathStr = "Images/{0}/".format(className)
output_file = Path(pathStr)
print(output_file, "output_file")
if not os.path.exists(pathStr):
    output_file.parent.mkdir(exist_ok=True, parents=True)

with device.running():
    # Use the factory calibration to undistort the depth frame and register the RGB
    # frame onto it.
    while True:
        for type_, frame in device:
            frames[type_] = frame
            if FrameType.Color in frames and FrameType.Depth in frames:
                break
        clr, dpth = frames[FrameType.Color], frames[FrameType.Depth]
        _, _, big_depth = device.registration.apply(clr, dpth, with_big_depth=True)
        ar = device.registration.get_big_points_xyz_array(big_depth)
        image = clr.to_array()
        cv2.imshow("img", image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            with open('Images/{0}/{1}.npy'.format(className, fileNumber), 'wb') as f:
                np.save(f, image)
                np.save(f, ar)
                fileNumber += 1
        print(fileNumber)
        device.get_next_frame()
        # time.sleep(1)
device.close()

numberOfSamples = fileNumber
dictionary = {
    "numberOfSamples": numberOfSamples
}

pathStr = "Images/{0}/info.json".format(className)
output_file = Path(pathStr)
output_file.parent.mkdir(exist_ok=True, parents=True)
with open(pathStr, "w") as outfile:
    json.dump(dictionary, outfile)
