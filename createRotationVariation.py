import cv2 as cv
import numpy as np
import os
import shutil
import random
import math
import shutil
import asyncio
import json


def rotate_image(image, angle):
    height, width = image.shape[:2]
    ##
    diagonal = int(math.ceil(math.sqrt(height**2 + width**2)))

    delta_w = diagonal - width
    delta_h = diagonal - height
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    image = cv.copyMakeBorder(
        image, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))

    image_center = diagonal // 2, diagonal // 2
    print(image_center)
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1)

    rotated = cv.warpAffine(image, rotation_mat,
                            (diagonal, diagonal), borderValue=(0, 0, 0, 0))

    return rotated


def trim(img):
    # get the center of the image
    height, width = img.shape[:2]
    center_x = width // 2
    center_y = height // 2
    # get top bound
    for i in range(center_y):
        if np.any(img[i, :] != 0):
            top = i
            break
    # get bottom bound
    for i in range(height - 1, center_y, -1):
        if np.any(img[i, :] != 0):
            bottom = i
            break
    # get left bound
    for i in range(center_x):
        if np.any(img[:, i] != 0):
            left = i
            break
    # get right bound
    for i in range(width - 1, center_x, -1):
        if np.any(img[:, i] != 0):
            right = i
            break
    return img[top:bottom, left:right]


def rotate_and_trim(img, angle):
    return trim(rotate_image(img, angle))


def create_variation(path, seed=0):
    raster_path = f"{path}/raster"
    position_path = f"{path}/positions.json"
    random.seed(seed)
    positions_json = None
    with open(position_path, 'r') as f:
        positions_json = json.load(f)

    for file in os.listdir(raster_path):
        img = cv.imread(f"{raster_path}/{file}", cv.IMREAD_UNCHANGED)
        angle = random.randint(0, 360)
        index = int(file.split('.')[0])
        img = rotate_and_trim(img, angle)
        positions_json['positions'][index * 3 + 2] = angle
        cv.imwrite('test.png', img)
        cv.imwrite(f"{raster_path}/{file}", img)
    with open(position_path, 'w') as f:
        json.dump(positions_json, f)


async def create_variations_helper(puzzle, semaphore):
    async with semaphore:
        # loop from a to z
        original_path = f"puzzle/{puzzle}"

        for i in range(65, 91):
            puzzle_name = f'{puzzle}-{chr(i)}'
            variant_path = f"puzzle/{puzzle_name}"
            # delete folder if exists
            shutil.rmtree(f"{variant_path}", ignore_errors=True)

            os.makedirs(f"{variant_path}")

            shutil.copytree(f"{original_path}/raster",
                            f"{variant_path}/raster")

            shutil.copy(f"{original_path}/positions.json",
                        f"{variant_path}/positions.json")
            create_variation(variant_path, seed=i)


async def create_variations():
    puzzles = os.listdir('puzzle')
    tasks = []
    cpu_count = os.cpu_count()
    semaphore = asyncio.Semaphore(cpu_count)

    for puzzle in puzzles:
        if 65 <= ord(puzzle[-1]) < 91:
            continue
        tasks.append(create_variations_helper(puzzle, semaphore))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(create_variations())
