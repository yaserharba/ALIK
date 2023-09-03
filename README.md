
# ALIK
ALIK (or Auto Labeling Images with Kinect) is a vision algorithm that can help us to do the labeling images step - one of the object detection AI training steps - automatically.

We use Kinect V2 to take the images, then we use [Yolov7](https://github.com/WongKinYiu/yolov7) to test and train the data.

## Installation and Prerequisites:

 1. OpenCV
 2. NumPy
 3. freenect2 from [here](https://rjw57.github.io/freenect2-python/)
 4. After installing freenect2, got to the [init](https://github.com/rjw57/freenect2-python/blob/master/freenect2/__init__.py) file in the installation directory and change this from:
```python
class QueueFrameListener(object):
    def __init__(self, maxsize=16):
        self.queue = Queue(maxsize=maxsize)
```
to:
```python
class QueueFrameListener(object):
    def __init__(self, maxsize=1):
        self.queue = Queue(maxsize=maxsize)
```
