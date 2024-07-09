import cv2
from sklearn import datasets

digits = datasets.load_digits()

print(digits)
# 0 < X < 9
X = 8
print(f"Soy la inteligencia artificial y he detectado que el dígito ingresado corresponde al número {X}")



