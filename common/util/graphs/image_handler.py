import os
import random
import sys

import cv2
import numpy as np

from business_entities.image import Image
from business_entities.image_train_row import ImageTrainRow
from common.util.logging.light_logger import LightLogger

class ImageHandler():

    def __init__(self):
        pass

    def __vectorize_images__(self,image_rows):
        train_x = []
        image_idx = []
        train_y = []

        for image_row in image_rows:
            train_x.append(image_row.X)
            train_y.append(image_row.Y)
            image_idx.append(image_row.image)

        #We convert them to num py
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        image_idx = np.array(image_idx)

        return  train_x,train_y,image_idx

    def __load_images__(self,true_images,false_images,true_label,offset,step_size):
        images = []
        i=0
        start = offset*step_size
        for true_img in true_images:
            if i>=start and i< (start+step_size):
                LightLogger.do_log("Adding TRUE img {} to the list which is a {}".format(true_img, true_label))
                images.append([true_img, true_label, 1])

            elif i>=(start+step_size):
                break

            i += 1

        i=0
        for false_img in false_images:
            if i >= start and i < (start + step_size):
                LightLogger.do_log("Adding FALSE img {} to the list which is a {}".format(false_img, "not" + true_label))
                images.append([false_img, "not " + true_label, 0])
            elif i >= (start + step_size):
                break

            i += 1

        return images

    def __load_image_rows__(self,images,def_width=500,def_height=375,with_reshape=True):
        image_rows = []

        for image in images:

            LightLogger.do_log("Processing img {}: Extracting pixels".format(image[0]))
            image_arr = cv2.imread(image[0])  # 0 containes the path
            image_arr = cv2.resize(image_arr, (def_width, def_height))

            image_classif = Image(image[0], image[1], image[1] + " pictures", image_arr, True if image[2] is 1 else False)  # 1 is the label title

            if with_reshape:
                image_arr = image_arr.reshape(-1)  # this FLATTENGS the width x height x 3 image!

            train_instance = ImageTrainRow(image_arr, image[2], image_classif)
            image_rows.append(train_instance)

        return  image_rows

    def __find_inner_files__(self,path,image_type):
        type_files = []
        for root_dir, dirs, files in os.walk(path.replace('"','')):
            for file in files:
                if file.endswith(image_type):
                    type_files.append(os.path.join(root_dir, file))
        return type_files

    def create_sets(self,true_path,false_path,true_label,image_type=".jpg",offset=0,step_size=2000):

        true_images= self.__find_inner_files__(true_path,image_type)
        false_images= self.__find_inner_files__(false_path,image_type)

        images=self.__load_images__(true_images,false_images,true_label,offset,step_size)

        image_rows = self.__load_image_rows__(images)

        LightLogger.do_log("Reshuffling {} instances".format(len(image_rows)))
        random.shuffle(image_rows)

        LightLogger.do_log("Properly vectorizing all the images")
        train_x,train_y,image_idx=self.__vectorize_images__(image_rows)

        LightLogger.do_log("Normalizing all the train_Xs")
        train_x=train_x/255

        train_x=train_x.transpose()
        train_y=train_y.reshape(1,len(train_y))

        LightLogger.do_log("Finally having a train_x vector with shape {} and train_y with shape {} ".format(train_x.shape,train_y.shape))

        # images have to go out READY to be consumed from the neural network
        #tain_x.shape=(widthXrowsX3,m)
        #train_y.shape(1,m)
        return  train_x,train_y,image_idx


    def create_non_vect_sets(self,true_path,false_path,true_label,image_type=".jpg"):
        true_images = self.__find_inner_files__(true_path, image_type)
        false_images = self.__find_inner_files__(false_path, image_type)

        images = self.__load_images__(true_images, false_images, true_label, 0, sys.maxsize)

        image_rows = self.__load_image_rows__(images,with_reshape=False)

        LightLogger.do_log("Reshuffling {} instances".format(len(image_rows)))
        random.shuffle(image_rows)

        LightLogger.do_log("Properly vectorizing all the images")
        train_x, train_y, image_idx = self.__vectorize_images__(image_rows)

        LightLogger.do_log("Normalizing all the train_Xs")
        train_x = train_x / 255

        return  train_x,train_y,image_idx




