import cv2
import numpy as np

m = [0.59607846, 0.60000002, 0.59607846, 0.60000002]
mean = 0.5980392396450043
sigma_sqr = 3.844674666630965e-06
res = [1 + (i - mean) * (0.60000002 - mean) / sigma_sqr for i in m]

print(res)
res = res[:3]
sm = sum(res)
res = [i / sm for i in res]
print(res)

array = np.array([0.0012694, 0.99746119, 0.0012694])
print(array[0])
array = array / sum(array)
a = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
print(cv2.cvtColor(a, cv2.COLOR_BGR2YUV))