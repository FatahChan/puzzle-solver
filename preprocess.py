import os
import asyncio
import cv2 as cv
import numpy as np


def resize_with_padding(image, desired_size, pad_value=0, value=(0, 0, 0), factor=None):
    if factor is not None:
        dim = (int(image.shape[1] * factor), int(image.shape[0] * factor))
        resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)

        delta_w = desired_size - resized.shape[1]
        delta_h = desired_size - resized.shape[0] 
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        padded = cv.copyMakeBorder(
            resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=value)
        return padded

    else:
        desired_size = desired_size - pad_value
        old_size = image.shape[:2]
        ratio = min(float(desired_size) /
                    old_size[0], float(desired_size) / old_size[1])
        new_size = tuple([int(x * ratio) for x in old_size])
        resized = cv.resize(image, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1] + pad_value
        delta_h = desired_size - new_size[0] + pad_value
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        padded = cv.copyMakeBorder(
            resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=value)

    return padded


def transparent_to_white(img, color=[255, 255, 255, 255]):
    if img.ndim != 3:
        return img
    img[np.where((img == [0, 0, 0, 0]).all(axis=2))] = color
    return img


def get_mask(img: np.ndarray):
    if img.ndim != 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv.threshold(gray, 254, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Create a black image as a mask
    mask = np.zeros_like(gray)

    # Draw contours on the mask
    cv.drawContours(mask, contours, 1, (255, 255, 255), thickness=20)

    return mask


def get_mask2(img: np.ndarray, size=128, padding=10, thickness=3):
    # RGBA
    _, binary = cv.threshold(img, 254, 255, cv.THRESH_BINARY)
    binary_resized = resize_with_padding(binary, size, padding, (0, 0, 0, 0))

    binary_resized = transparent_to_white(binary_resized)

    binary_resized_gray = cv.cvtColor(binary_resized, cv.COLOR_RGBA2GRAY)

    countours, _ = cv.findContours(
        binary_resized_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(binary_resized)
    idx = 0
    for i in range(len(countours)):
        if len(countours[i]) > len(countours[idx]):
            idx = i

    cv.drawContours(canvas, countours, idx, (255, 255, 255, 255), thickness)
    canvas_gray = cv.cvtColor(canvas, cv.COLOR_RGB2GRAY)

    return canvas_gray


def get_contours(img, mask):
    masked = np.copy(img)
    white_pixels = np.logical_and(
        mask == 255, np.any(img != [255, 255, 255], axis=-1))
    masked[white_pixels] = img[white_pixels]
    masked[~white_pixels] = [255, 255, 255]
    return masked


def get_mask3(img, size, padding, thickness, outline_color, factor=None):
    # RGBA
    mask = np.zeros_like(img)
    mask = resize_with_padding(
        mask, size, padding, (0, 0, 0, 0), factor=factor)
    image_resize = resize_with_padding(
        img, size, padding, (0, 0, 0, 0), factor=factor)
    # if not transparent, then black
    mask[np.where((image_resize == [0, 0, 0, 0]).all(axis=2))] = [0, 0, 0, 255]
    # if not black then white
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j][3] != 255:
                mask[i][j] = [255, 255, 255, 255]

    mask = cv.cvtColor(mask, cv.COLOR_RGBA2GRAY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    outline = np.zeros_like(cv.cvtColor(mask, cv.COLOR_GRAY2RGB))
    idx = 0
    for i in range(len(contours)):
        if len(contours[i]) > len(contours[idx]):
            idx = i

    cv.drawContours(mask, contours, idx, 255, thickness)
    cv.drawContours(outline, contours, idx, (255, 0, 0), 1)

    return mask, outline


def draw_outline(img, outline):
    img = img.copy()
    # if pixel is not black in outline, then draw it on img
    for i in range(len(outline)):
        for j in range(len(outline[i])):
            if not (outline[i][j][0] == 0 and outline[i][j][1] == 0 and outline[i][j][2] == 0):
                img[i][j] = outline[i][j]
    return img


def preprocess(img, factor=None):
    size = 128
    padding = 10
    thickness = 10

    new_image = img.copy()
    new_image = transparent_to_white(new_image)

    new_image = resize_with_padding(
        new_image, size, padding, (255, 255, 255, 255), factor=factor)

    new_image = cv.cvtColor(new_image, cv.COLOR_RGBA2RGB)
    mask, outline = get_mask3(
        img.copy(), size, padding, thickness, (255, 0, 0), factor=factor)
    new_image = get_contours(new_image, mask)
    new_image = draw_outline(new_image, outline)
    cv.imwrite("test.png", new_image)
    return new_image

async def create_preproccessed(parent_path, file, semaphore, Factor=None):
    async with semaphore:
        print(f"Processing {parent_path}/raster/{file}")
        img = cv.imread(f"{parent_path}/raster/{file}", cv.IMREAD_UNCHANGED)
        img = preprocess(img, factor=Factor)
        print(img.shape)
        print(
            f"Saving {parent_path}/preprocessed/{file.replace('png', 'jpg')}")
        cv.imwrite(
            f"{parent_path}/preprocessed/{file}", img)
        cv.imwrite("test.png", img)


def getFactor(path):
    padding = 10
    max_width = 0
    max_height = 0
    for i in os.listdir(path):
        cv_img = cv.imread(path + i, cv.IMREAD_UNCHANGED)
        if cv_img.shape[0] > max_height:
            max_height = cv_img.shape[0]
        if cv_img.shape[1] > max_width:
            max_width = cv_img.shape[1]

    return (128 - padding) / max(max_width, max_height)


async def main():
    task = []
    cpu_count = os.cpu_count()
    semaphore = asyncio.Semaphore(cpu_count)
    lenght = len(os.listdir("puzzle"))
    count = 0
    for dir in os.listdir("puzzle"):

        if dir == ".DS_Store":
            continue

        factor = getFactor(f"puzzle/{dir}/raster/")
        print(factor)
        print("prog",count/lenght * 100)
        count += 1
        for file in os.listdir(f"puzzle/{dir}/raster"):
            if file == ".DS_Store":
                continue
            os.makedirs(f"puzzle/{dir}/preprocessed", exist_ok=True)
            task.append(create_preproccessed(
                f"puzzle/{dir}", file, semaphore, factor))

    await asyncio.gather(*task)


if __name__ == "__main__":
    asyncio.run(main())
