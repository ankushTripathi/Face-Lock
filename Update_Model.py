__author__ = 'ankush'

import cv2
import glob
import numpy as np


lbphFace = cv2.createLBPHFaceRecognizer()

def make_sets(people):
    training_data = []
    training_labels = []

    for person in people:
        training = glob.glob("dataset\\%s\\*" % person)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(people.index(person))

    return training_data, training_labels


def run_recognizer(people):
    training_data, training_labels = make_sets(people)
    print("training fisher Face classifier...")
    print("size of training set is :" + str(len(training_labels)) + " images")
    lbphFace.train(training_data, np.asarray(training_labels))
    print("model trained!")

def update(people):
    run_recognizer(people)
    print("saving model...")
    lbphFace.save("xml/faces.xml")
    print("model saved!")
