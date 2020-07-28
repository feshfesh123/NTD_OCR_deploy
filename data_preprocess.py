import os
from net_config import ArchitectureConfig, FilePaths
import cv2
import numpy as np
from shutil import copyfile
from PIL import Image
import random

def remove_invalid_data(root = FilePaths.fnDataset):
    print("Data Error...")
    print(root)
    file_list = os.listdir(root)
    chars = ArchitectureConfig.CHARS
    count = 0
    d = dict()
    for file_name in file_list:
        if file_name.endswith(".txt"):
            label_name = os.path.join(root, file_name)
            file_image = file_name.replace("txt", "jpg")
            image_name = os.path.join(root, file_image)
            with open(label_name, encoding="utf-8-sig") as f:
                lines = f.readlines()
                word = lines[0]
                for ch in list(word):
                    if (chars.count(ch) == 0):
                        if (ch in d):
                            d[ch]+=1
                        else:
                          d[ch] = 0
                        os.remove(label_name)
                        os.remove(image_name)
                        count+=1
                        break
    print(d)
    print("Removed ", count," invalid datas in ", root, " !!!")

BINARY_THREHOLD = 180

def process_image_for_ocr(file_path, des_file_path):
    scale_img(file_path, des_file_path)
    im_new = skew_correction(des_file_path)
    #im_new = remove_noise_and_smooth(des_file_path)
    return im_new

def scale_img(file_path, des_file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # (h, w)
    (wt, ht) = ArchitectureConfig.IMG_SIZE
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    resized_image = cv2.resize(img, newSize)  # (h, w)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = resized_image
    cv2.imwrite(des_file_path, target)

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                     3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    cv2.imwrite(file_name, or_image)
    return or_image

def skew_correction(file_name):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)
    (cx,cy), (w,h), ang = ret
    if w<h:
        w,h = h,w
        ang += 90
    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = 255 - cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
    img = Image.fromarray(rotated,'L')
    img = img.save(file_name)
    return img 

def preprocess_all_data():
    i = 1
    for root in FilePaths.fnDataCollection:
        # Remove data invalid
        remove_invalid_data(root)
        # Preprocess data
        file_list = os.listdir(root)
        for file_name in file_list:
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                # Tao duong dan
                file_path = os.path.join(root, file_name)
                des_file_path = os.path.join(FilePaths.fnDataPreProcessed, str(i) + '.jpg')
                # Xu ly image
                process_image_for_ocr(file_path, des_file_path)
                # Sao chep label
                label_path = file_path.replace(".png", ".txt")
                label_path = label_path.replace(".jpg", ".txt")
                copyfile(label_path, des_file_path.replace(".jpg", ".txt"))
                i+=1
    print("Preprocessed ", i-1, " images !")

# #process_image_for_ocr("ocr.jpg", "ocr-test.jpg")
# img = cv2.imread("ocr.jpg")
# stretch = (random.random() - 0.5)  # -0.5 .. +0.5
# wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
# img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5
# cv2.imshow("", img)
# cv2.waitKey()