import random
import numpy as np
import os, sys
from scipy import ndimage
from skimage.io import imread
from six.moves import cPickle as pickle
from cv2 import resize
import json
import argparse

# model architecture from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#np.random.seed(123)  # for reproducibility
#labels will be in order which is printed as "dirs found" in function "maybe_pickle"

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('tf')

from PIL import Image

class Classificator:

    #make arrays to hold datasets of all three classes
    def make_arrays(self, nb_rows, image_height, image_width, channels):
      if nb_rows:
        dataset = np.ndarray((nb_rows, image_height, image_width, channels), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
      else:
        dataset, labels = None, None
      return dataset, labels

    # Load the data for a single country label to numpy array, normalization to [-1,1]
    def load_country(self, folder): # e.g. folder=train_dir/GER
        image_paths = os.listdir(folder)
        dataset = np.ndarray(shape=(len(image_paths), self.image_height, self.image_width, 3), dtype=np.float32)
        num_images = 0
        for path in image_paths:
            image_file = os.path.join(folder, path)
            try:
                image_data = (ndimage.imread(image_file).astype(float)) / self.pixel_depth
                image_data = image_data[:,:]
                if image_data.shape != (self.image_height, self.image_width, 3):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[num_images, :, :, :] = image_data
                num_images = num_images + 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

        dataset = dataset[0:num_images, :, :, :]

        print('Full dataset tensor:', dataset.shape)
        print('Mean:', np.mean(dataset))
        print('Standard deviation:', np.std(dataset))
        return dataset

    def maybe_pickle(self, data_folders, dataset, force=False):
        dirs_in_train = os.listdir(data_folders)
        print("dirs found: ", dirs_in_train)
        print("dataset: ", dataset)
        dataset_names = []
        for folder in dirs_in_train: #for each country in train
            set_filename = '{}_{}.pickle'.format(dataset, folder) # e.g."train_GB.pickle"
            dataset_names.append(set_filename)
            pickle_path = os.path.join(self.root_dir, set_filename) #save to class_dir
            if os.path.exists(pickle_path) and not force: #You may override by setting force=True.
                print('%s already present - Skipping pickling.' % pickle_path)
            else:
                print('Pickling %s' % pickle_path)
                print("using folder: ", folder)
                path = os.path.join(data_folders, folder)
                print("path: ", path)
                print("pickle_path: ", pickle_path)
                data = self.load_country(path)
                try:
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', pickle_path, ':', e)
        return dataset_names

    # 3) merge datasets to hold all classes
    def merge_datasets(self, pickle_files, train_size, test_size=0):
        num_classes = self.classes
        test_dataset, test_labels = self.make_arrays(test_size, self.image_height, self.image_width, 3)
        train_dataset, train_labels = self.make_arrays(train_size, self.image_height, self.image_width, 3)
        vsize_per_class = test_size // num_classes
        tsize_per_class = train_size // num_classes
        start_v, start_t = 0, 0
        end_v, end_t = vsize_per_class, tsize_per_class
        end_l = vsize_per_class + tsize_per_class

        for label, pickle_file in enumerate(pickle_files):  # train_GB.pickle, train_GER.pickle...
            try:
                with open(pickle_file, 'rb') as f:
                    country_set = pickle.load(f)
                    # let's shuffle the letters to have random validation and training set
                    np.random.shuffle(country_set)
                    if test_dataset is not None:
                        test_country = country_set[:vsize_per_class, :, :]
                        test_dataset[start_v:end_v, :, :] = test_country
                        test_labels[start_v:end_v] = label
                        #print("label: ", label)
                        start_v += vsize_per_class
                        end_v += vsize_per_class

                    train_country = country_set[vsize_per_class:end_l, :, :]
                    train_dataset[start_t:end_t, :, :] = train_country
                    train_labels[start_t:end_t] = label
                    start_t += tsize_per_class
                    end_t += tsize_per_class
            except Exception as e:
                print('Unable to process data from', pickle_file, ':', e)
                raise

        return test_dataset, test_labels, train_dataset, train_labels

    # 4) shuffle datasets, so that classes are not in order
    def randomize(self, dataset, labels):
      permutation = np.random.permutation(labels.shape[0])
      shuffled_dataset = dataset[permutation,:,:]
      shuffled_labels = labels[permutation]
      return shuffled_dataset, shuffled_labels

    def train_model(self, X_train, Y_train, X_test, Y_test):
        # model architecture   64x16
        model = Sequential()

        model.add(Convolution2D(16, (3, 3), padding="same", activation='relu', input_shape=(self.image_height, self.image_width, 3)))
        model.add(Convolution2D(16, (3, 3), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, (3, 3), padding="same", activation='relu'))
        model.add(Convolution2D(32, (3, 3), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes, activation='softmax'))  # first argument = no. of classes

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=self.epochs, verbose=1)

        model.save_weights(self.weights_path)
        with open(self.model_path, 'w') as json_out:
            json_model = model.to_json()
            json.dump(json_model, json_out)

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
                max_ix = int(np.argmax(pred, 1))
                output_class = self.pretty_labels[max_ix]
                #json_output = {"output_class": output_class, "probability": float(probability)}
                #json_out = json.dumps(json_output)
                return output_class #json_out
            else:
                return "default"
        else:
            return None

    # returns most likely country, and its probability
    #def get_json_prediction(self, prediction):
    #    probability = np.amax(prediction)
    #    max_ix = int(np.argmax(prediction, 1))
    #    output_class = self.pretty_labels[max_ix]
    #    json_output = {"output_class": output_class, "probability": float(probability)}
    #    json_out = json.dumps(json_output)
    #    return json_out

    def get_pretty_labels(self, class_dir):
        train_dir = os.path.join(class_dir, "train")
        dirs = os.listdir(train_dir)
        return dirs

    def train_eval_pred(self, X_train, Y_train, X_test, Y_test, X_valid, Y_valid):
        self.train_model( X_train, Y_train, X_test, Y_test)
        model = self.get_loaded_model(self.model_path, self.weights_path)
        print("evaluation: ")
        print(model.evaluate(X_valid, Y_valid, verbose=1))
        self.predict_classes(model, X_test)

    def create_and_train_datasets(self, class_dir):
        train_dir = os.path.join(class_dir, "train")
        valid_dir = os.path.join(class_dir, "valid")

        # 1+2) load images to numpy-array, save to pickle file
        #i.e. returns 6 files: [train/test]_[GB/GER/NL].pickle
        train_datasets = self.maybe_pickle(train_dir, "train")
        valid_datasets = self.maybe_pickle(valid_dir, "valid")
        # make sure you are in classification-directory
        os.chdir(class_dir)
        print("cwd: ", os.getcwd())

        # create three datasets (train, valid, test) with corresponding labels
        test_dataset, test_labels, train_dataset, train_labels = self.merge_datasets(train_datasets, self.train_size, self.test_size)
        _, _, valid_dataset, valid_labels = self.merge_datasets(valid_datasets, self.valid_size)

        # check shape of datasets
        print('Training:', train_dataset.shape, train_labels.shape)
        print('Testing:', test_dataset.shape, test_labels.shape)
        print('Validation:', valid_dataset.shape, valid_labels.shape)

        train_dataset, train_labels = self.randomize(train_dataset, train_labels)
        test_dataset, test_labels = self.randomize(test_dataset, test_labels)
        valid_dataset, valid_labels = self.randomize(valid_dataset, valid_labels)

        # assign datasets
        # run maybe_pickle() before this step, also works if only pickle files are present in classification-directory
        X_train = train_dataset
        Y_train = train_labels
        X_test = test_dataset
        Y_test = test_labels
        X_valid = valid_dataset
        Y_valid = valid_labels

        # Converts a class vector (integers) to binary class matrix.
        print("Y_train: ", Y_train)
        print("Y_test: ",Y_test)
        print("Y_valid: ", Y_valid)
        Y_train = np_utils.to_categorical(Y_train, self.classes)
        Y_test = np_utils.to_categorical(Y_test, self.classes)
        Y_valid = np_utils.to_categorical(Y_valid, self.classes)
        print("labels shape: ", Y_valid.shape)

        self.train_eval_pred(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, X_valid=X_valid, Y_valid=Y_valid)

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
        parser.add_argument('-tr', '--train_size', required=False, default=0, type=int, help="number of images per class in train dataset")
        parser.add_argument('-te', '--test_size', required = False, default=0, type=int, help="number of images per class in test dataset")
        parser.add_argument('-va', '--valid_size', required=False, default=0, type=int, help="number of images per class in validation dataset")
        parser.add_argument('-w', '--image_width', required=False, default=64, type=int, help='width of input images')
        parser.add_argument('-he', '--image_height', required=False, default=16, type=int, help='height of input images')
        parser.add_argument('-e', '--epochs', required=False, default=10, type=int, help='no. of epochs for training')
        opt = parser.parse_args()

        self.mode = opt.mode
        self.root_dir = opt.root_dir
        self.model_path = opt.model
        self.weights_path = opt.weights

        self.image_width = opt.image_width
        self.image_height = opt.image_height
        self.pixel_depth = 255.

        self.classes = int(opt.classes)
        self.train_size = opt.train_size * self.classes  # 8000
        self.test_size = opt.test_size * self.classes  # 2400
        self.valid_size = opt.valid_size * self.classes  # 2400

        self.epochs = opt.epochs
        self.pretty_labels = self.get_pretty_labels(self.root_dir)

        self.test_img = os.path.join(self.root_dir, opt.test_img)

        if self.mode=="train":
            print("start create_and_train")
            self.create_and_train_datasets(self.root_dir)

        if self.mode=="pred":
            #print("start prediction")
            image = (imread(self.test_img).astype(np.float32)) / self.pixel_depth
            image = resize(image, (self.image_height, self.image_width))
            image = np.reshape(image, [1, self.image_height, self.image_width, 3])

            self.weights_path = os.path.join(self.root_dir, self.weights_path)
            self.model_path = os.path.join(self.root_dir, self.model_path)
            model = self.get_loaded_model(self.model_path, self.weights_path)
            pred_json = self.predict_classes(model, image)
            #sys.exit(pred_json)
            sys.stdout.write(pred_json)
            sys.exit(0)
            #print("json: ", pred_json)

        if self.mode=="test":
            print("weights path exists: ", os.path.exists(self.weights_path))
            os.mknod(self.weights_path)

if __name__ == '__main__':
    Classificator()
