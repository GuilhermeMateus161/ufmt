import cv2
import mediapipe as mp
import time
import pathlib
import numpy as np
from mtcnn import MTCNN

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"


class FaceDetector():
    def __init__(self, conf = 0.55):

        self.conf = conf  #confiança minima da detecção

        self.detector = mp.solutions.face_detection     #instancia o detector
        self.detector = self.detector.FaceDetection(self.conf)    #informa a confiança

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #muda o formato da imagem GBR para RGB
        self.results = self.detector.process(imgRGB)    #detector processando a imagem 

        bboxs = [] 

        if self.results.detections:    #se tiver detecções
            for id, detection in enumerate(self.results.detections):

                bboxC = detection.location_data.relative_bounding_box   #pegando informações do bounding box
                ih, iw, ic = img.shape     #pegando dimensões da imagem

                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)    #criando bounding box
                
                bboxs.append([id, bbox, detection.score])   #adicionando dentro de uma lista para casos com mais de uma face

                if draw:
                    img = self.drawBound(img,bbox)

                    cv2.putText(img, f'mediapipe',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 0, 255), 2)
        return img, bboxs

    def drawBound(self, img, bbox, l=30, t=2, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)

        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = FaceDetector()
    cascade_detector = cv2.CascadeClassifier(str(cascade_path))
    detector_mtcnn = MTCNN()

    key = ord('1')

    while True:
        _, img = cap.read()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


        if key == 27 or key == ord('2') or key == ord('1') or key == ord('3'):
            last_key = key
             
        key = cv2.waitKey(1)

        if last_key == 27:
            break

        if last_key == ord('1'):
            img, bboxs = detector.findFaces(img)

        if last_key == ord('2'):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            faces = cascade_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,0), 2)
                cv2.putText(img, 'cascade', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

        if last_key == ord('3'):
            img_detec = np.asarray(img)
            results = detector_mtcnn.detect_faces(img_detec)

            if results:
                x, y, w, h = results[0]['box']

                cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(img, 'mtcnn', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            
        cv2.imshow("Image", img)


if __name__ == "__main__":
    main()