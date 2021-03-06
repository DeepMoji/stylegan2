# from https://github.com/rolux

import os
import sys
import bz2
from tensorflow.keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

import logging

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    # Uncomment to download
    # landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
    #                                            LANDMARKS_MODEL_URL, cache_subdir='temp'))
    RAW_IMAGES_DIR = sys.argv[1]
    ALIGNED_IMAGES_DIR = sys.argv[2]

    log_data = False
    if len(sys.argv) > 3:
        LOG_FILE = sys.argv[3]
        log_data = True

    if log_data:
        logging.basicConfig(level=logging.INFO, filename=LOG_FILE)

    # landmarks_detector = LandmarksDetector(landmarks_model_path)
    landmarks_detector = LandmarksDetector('/Users/michaelko/Code/backup/lib/shape_predictor_68_face_landmarks.dat')
    cnt = -1
    file_list = [x for x in os.listdir(RAW_IMAGES_DIR) if x[0] not in '._']
    file_list.sort()
    if log_data:
        logging.info(' There are ' + str(len(file_list)) + ' files')
    for img_name in file_list:
        cnt = cnt + 1
        res_string = "Processing image num " + str(cnt) + " " + img_name
        if log_data:
            logging.info(res_string)
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        num_landmarks = 0
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks)
            num_landmarks = num_landmarks + 1
        if log_data:
            logging.info('There are ' + str(num_landmarks) + ' landmarks')
    if log_data:
        logging.info('Done processing')
