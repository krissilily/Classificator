#import random
import numpy as np
import os, sys
#from scipy import ndimage
from skimage.io import imread
#from six.moves import cPickle as pickle
from cv2 import resize
import json
import argparse

# model architecture from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#np.random.seed(123)  # for reproducibility
#labels will be in order which is printed as "dirs found" in function "maybe_pickle"

from keras.models import Sequential, load_model, model_from_json
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('tf')

from PIL import Image

class Classificator:

    #returns compiled model from json-model and hdf5-weights
    def get_loaded_model(self, model_path, weights_path):
        with open(model_path, 'r') as json_file:
            loaded_model_json = json.load(json_file)
            model = model_from_json(loaded_model_json)
            model.load_weights(weights_path)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # returns prediction array, and label of most likely country label
    def predict_classes(self, model, X_test):
        pred = model.predict(X_test, batch_size=1)
        #print("all predictions: ")
        #print(np.around(pred, 2))
        if pred.shape[0] == 1: # only calculate most likely country if prediction input was single image
            probability = np.amax(pred)
            if probability > 0.7:
                max_ix = str(int(np.argmax(pred, 1)))
                json_output = {"output_class": max_ix, "probability": float(probability)}
                json_out = json.dumps(json_output)
                return json_out#max_ix#output_class #json_out
            else:
                return "default"
        else:
            return None

    # class_dir needs to contain directories "train" and "valid"
    # train_size, test_size, valid_size need to be adjusted according to number of images used for training:
    # train + test images should be in the "train" directory
    # example: 1000 images per class for train, 300 for test, 300 for validation -> 1300 images in train, 300 in valid -> but train_size=8000, test_size=2400, valid_size=2400 for 8 classes
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mode', default='train', required=True, help='train or pred')
        parser.add_argument('-d', '--root_dir', required=True, help='path to image directory')
        parser.add_argument('--model', required=True, help='model file name, must lie in class_dir')
        parser.add_argument('--weights', required=True, help='weights file name, must lie in class_dir')
        parser.add_argument('-i', '--test_img', required=False, default="", help='image file name, must lie in class_dir')
        parser.add_argument('-c', '--classes', required=False, type=int, default=3, help='number of classes')
        parser.add_argument('-w', '--image_width', required=False, default=64, type=int, help='width of input images')
        parser.add_argument('-he', '--image_height', required=False, default=16, type=int, help='height of input images')
        opt = parser.parse_args()

        self.mode = opt.mode
        self.root_dir = opt.root_dir
        self.model_path = opt.model
        self.weights_path = opt.weights

        self.image_width = opt.image_width
        self.image_height = opt.image_height

        self.classes = int(opt.classes)
        #self.pretty_labels = self.get_pretty_labels(self.root_dir)

        self.test_img = os.path.join(self.root_dir, opt.test_img)

        if self.mode=="train":
            print("start create_and_train")
            self.create_and_train_datasets(self.root_dir)

        if self.mode=="pred":
            #print("start prediction")
            image = (imread(self.test_img).astype(np.float32)) / 255.
            image = resize(image, (self.image_height, self.image_width))
            image = np.reshape(image, [1, self.image_height, self.image_width, 3])

            self.weights_path = os.path.join(self.root_dir, self.weights_path)
            self.model_path = os.path.join(self.root_dir, self.model_path)
            model = self.get_loaded_model(self.model_path, self.weights_path)
            pred_json = self.predict_classes(model, image)
            #sys.exit(pred_json)
            sys.stdout.write(pred_json)
            sys.exit(0)
            print("json: ", pred_json)

        if self.mode=="test":
            print("weights path exists: ", os.path.exists(self.weights_path))
            os.mknod(self.weights_path)

if __name__ == '__main__':
    Classificator()
