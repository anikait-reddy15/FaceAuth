import os
import random
import cv2 
import numpy as np
import pandas as pd

class LFWTripletGenerator:
    def __init__(self, dataset_path=r"C:\Projects\FaceAuth\Model\Datasets\lfw-deepfunneled",
                  batch_size=32, target_shape=(105, 105)):
        '''
        dataset_path (str) : Path to the LFW image dataset
        batch_size (int) : The batch size to generate triplets in one iteration
        target_shape (tuple) : The image shape being resized to
        '''

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.people_dict = self.load_people_dict()
        self.people_names = list(self.people_dict.keys())

    def load_people_dict():
        '''
        Creates a dictonary that contains person name and their respective images
        people_dict = {"person_name" : ["image1.jpg", "image2.jpg" ....)}
        Filters people out with less than 2 images
        '''

        people_dict = {}
        

if __name__ == "__main__":
    os.listdir(r"C:\Projects\FaceAuth\Model\Datasets\lfw-deepfunneled")