import dlib
from imutils.face_utils import FaceAligner
import cv2
import os


class FaceAlignerV1:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./face_align_model/shape_predictor_68_face_landmarks.dat")
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=48)


if __name__ == '__main__':
    face_aligner = FaceAlignerV1()
    for parent, dir_name, filenames in os.walk('./Origin'):
        for filename in filenames:
            file_full_path = os.path.join(parent, filename)
            img = cv2.imread(file_full_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_aligner.detector(img, 1)
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                out = face_aligner.fa.align(img, gray, d)
                dest_path = parent.replace('./Origin', './Aligned')
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                cv2.imwrite(os.path.join(dest_path, filename), out)
