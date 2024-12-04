import cv2
import numpy as np
import os
import random
import json
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate a synthetic dataset of shapes.")
parser.add_argument("--output_dir", type=str, default="shapes_dataset", help="Directory to save the dataset")
parser.add_argument("--num_images", type=int, default=300, help="Number of images to generate")
parser.add_argument("--image_size", type=int, default=128, help="Size of each image (pixels)")
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Function to create an image with a random shape
def create_image(image_size=128, shape=None):
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    thickness = -1
    center = (random.randint(30, image_size - 30), random.randint(30, image_size - 30))
    size = random.randint(20, 40)

    bounding_box = None

    if shape == "circle":
        cv2.circle(img, center, size, color, thickness)
        bounding_box = [center[0] - size, center[1] - size, center[0] + size, center[1] + size]
    elif shape == "square":
        top_left = (center[0] - size, center[1] - size)
        bottom_right = (center[0] + size, center[1] + size)
        cv2.rectangle(img, top_left, bottom_right, color, thickness)
        bounding_box = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
    elif shape == "triangle":
        pt1 = (center[0], center[1] - size)
        pt2 = (center[0] - size, center[1] + size)
        pt3 = (center[0] + size, center[1] + size)
        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.drawContours(img, [triangle_cnt], 0, color, thickness)
        bounding_box = [min(pt1[0], pt2[0], pt3[0]), min(pt1[1], pt2[1], pt3[1]),
                max(pt1[0], pt2[0], pt3[0]), max(pt1[1], pt2[1], pt3[1])]

    return img, bounding_box

# Generate the dataset
shapes = ["circle", "square", "triangle"]
annotations = []

for i in range(args.num_images):
    shape = random.choice(shapes)
    img, bbox = create_image(image_size=args.image_size, shape=shape)
    filename = f"{shape}_{i}.png"
    cv2.imwrite(os.path.join(args.output_dir, filename), img)

    # Save annotations
    annotations.append({"filename": filename, "label": shape, "bounding_box": bbox})

    # Log progress every 50 images
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{args.num_images} images generated.")

# Save annotations to a JSON file
with open(os.path.join(args.output_dir, "annotations.json"), "w") as f:
    json.dump(annotations, f, indent=4)

print(f"Dataset generated in '{args.output_dir}'. Total images: {args.num_images}.")