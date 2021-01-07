
import bz2
from tensorflow.keras.utils import get_file
from ffhq_dataset.landmarks_detector import LandmarksDetector
from ffhq_dataset.face_alignment import image_align

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

if __name__ == "__main__":
    print('Start')

    # landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
    #                                            LANDMARKS_MODEL_URL, cache_subdir='temp'))

    landmarks_detector = LandmarksDetector('/Users/michaelko/Code/coreml-playground/lib/shape_predictor_68_face_landmarks.dat')

    face_landmarks = landmarks_detector.get_landmarks('/Users/michaelko/Downloads/miya.png')

    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks('/Users/michaelko/Downloads/miya.png'), start=1):
        image_align('/Users/michaelko/Downloads/miya.png', '/Users/michaelko/Downloads/miya1.png', face_landmarks)
