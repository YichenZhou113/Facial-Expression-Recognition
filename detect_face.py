import face_recognition
import argparse
import pickle
import cv2
import numpy as np

def detect(image):
    if image.shape[2] != 3:
        print('...')


    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,
    	model = 'cnn')
    #print(boxes)
    images = []
    for box in boxes:
        img = image[box[0] : box[2] , box[3] : box[1]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray)

    return images, boxes


def main():
    image = cv2.imread('test.jpg')
    faces = np.array(detect(image))
    print(faces.shape)

if __name__ == "__main__":
    main()
