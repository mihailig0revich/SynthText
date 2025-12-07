# transforms.py

import numpy as np
import cv2

def aug_rotate(img, masks, angle_deg):
    # Поворот изображения и масок
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle_deg, 1)
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    rotated_masks = [cv2.warpAffine(mask, M, (img.shape[1], img.shape[0])) for mask in masks]
    return rotated_img, rotated_masks

def aug_perspective(img, masks, max_jitter_ratio=0.08):
    # Применение случайной перспективы
    height, width = img.shape[:2]
    jitter = np.random.uniform(-max_jitter_ratio, max_jitter_ratio, size=8)  # Генерируем 8 случайных значений
    pts1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts2 = np.float32([[jitter[0], jitter[1]], [width + jitter[2], jitter[3]],
                       [width + jitter[4], height + jitter[5]], [jitter[6], height + jitter[7]]])  # Используем все 8 элементов
    M = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_img = cv2.warpPerspective(img, M, (width, height))
    perspective_masks = [cv2.warpPerspective(mask, M, (width, height)) for mask in masks]
    return perspective_img, perspective_masks

