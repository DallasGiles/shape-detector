import cv2
import numpy as np
import os
import random

os.makedirs("shapes_dataset", exist_ok=True)

def create_image(image_size=128, shape=None):

    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0,255))
    thickness = -1
    center = (random.randint(30, image_size-30), random.randint(30, image_size-30))
    size = random.randint(20, 40)

    if shape == "circle":
        cv2.circle(img, center, size, color, thickness)
    elif shape == "square":
        top_left = (center[0] - size, center[1] - size)
        bottom_right  = (center[0] + size, center[1] + size)
        cv2.rectangle(img, top_left, bottom_right, color, thickness)
    elif shape == "triangle":
        pt1 = (center[0], center[1] - size)
        pt2 = (center[0] - size, center[1] + size)
        pt3 = (center[0] + size, center[1] + size)
        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.drawContours(img, [triangle_cnt], 0, color, thickness)

    return img

shapes = ["circle", "square", "triangle"]
for i in range(300):
    shape = random.choice(shapes)
    img = create_image(shape=shape)
    cv2.imwrite(f"shapes_dataset/{shape}_{i}.png", img)