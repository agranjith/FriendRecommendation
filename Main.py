from pyexpat import model
from re import T
from select import select
from tabnanny import verbose
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input,decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras import layers,models
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Lambda, Dense, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import config
from glob import glob
import glob as gb

import scipy
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    

class LoadData:
    def __init__(self):
        pass
    def from_folder(self,folder_list:list):
        self.folder_list = folder_list
        image_path = []
        for folder in self.folder_list:
            for path in os.listdir(folder):
                image_path.append(os.path.join(folder,path))
        return image_path 
    def from_csv(self,csv_file_path:str,images_column_name:str):
        self.csv_file_path = csv_file_path
        self.images_column_name = images_column_name
        return pd.read_csv(self.csv_file_path)[self.images_column_name].to_list()

class FeatureExtractor:
    def __init__(self):
        self.base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('fc1').output)
    def extract(self, img):
        img = img.resize((224, 224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
    def get_feature(self,image_data:list):
        self.image_data = image_data 
        features = []
        for img_path in tqdm(self.image_data): 
            try:
                feature = self.extract(img=Image.open(img_path))
                features.append(feature)
            except:
                features.append(None)
                continue
        return features

    def results(self):
        tf.keras.applications.VGG16(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
        model = VGG16()
        
        plot_model(model, to_file='vgg_model.png')

        model.summary()

        filePath = "D:\Project\Final Year\Friend Rcommndation System\Final 2\\vggImagenet\\blurpexels-anthony-133394.jpg"

        image1 = image.load_img(filePath, target_size = (224, 224))

        transformedImage = image.img_to_array(image1)

        transformedImage = np.expand_dims(transformedImage, axis = 0)

        transformedImage = preprocess_input(transformedImage)

        prediction = model.predict(transformedImage)

        predictionLabel = decode_predictions(prediction, top = 5)
        print(predictionLabel)

        print('%s (%.2f%%)' % (predictionLabel[0][0][1], predictionLabel[0][0][2]*100 ))
    
    def cust_dataset(self):
        train_path = "D:\Project\Final Year\Friend Rcommndation System\Final 2\images\\fruit\\fruits-360-original-size\\fruits-360-original-size\Training"
        valid_path = "D:\Project\Final Year\Friend Rcommndation System\Final 2\images\\fruit\\fruits-360-original-size\\fruits-360-original-size\Validation"

        image_files = glob(train_path + '/*/*.jp*g')
        valid_image_files = glob(valid_path + '/*/*.jp*g')

        folders = glob(train_path + '/*')

        IMAGE_SIZE = [100, 100]

        epochs = 1
        batch_size = 32
        vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

        for layer in vgg.layers:
          layer.trainable = False

        x = Flatten()(vgg.output)
        prediction = Dense(len(folders), activation='softmax')(x)

        model = Model(inputs=vgg.input, outputs=prediction)

        model.summary()

        model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy']
        )

        gen = ImageDataGenerator(
          rotation_range=20,
          width_shift_range=0.1,
          height_shift_range=0.1,
          shear_range=0.1,
          zoom_range=0.2,
          horizontal_flip=True,
          vertical_flip=True,
          rescale=1./255,  
          preprocessing_function=preprocess_input
        )


        test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
        print(test_gen.class_indices)
        labels = [None] * len(test_gen.class_indices)
        for k, v in test_gen.class_indices.items():
          labels[v] = k
        train_generator = gen.flow_from_directory(
            train_path,
            target_size=IMAGE_SIZE,
            shuffle=True,
            batch_size=batch_size,
        )
        valid_generator = gen.flow_from_directory(
            valid_path,
            target_size=IMAGE_SIZE,
            shuffle=False,
            batch_size=batch_size,
        )

        r = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=epochs,
            steps_per_epoch=len(image_files) // batch_size,
            validation_steps=len(valid_image_files) // batch_size,
        )
        plt.plot(r.history['loss'], label='train loss')
        plt.plot(r.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()

        plt.plot(r.history['accuracy'], label='train acc')
        plt.plot(r.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show()

        print("Final training accuracy = {}".format(r.history["accuracy"][-1]))
        print("Final validation accuracy = {}".format(r.history["val_accuracy"][-1]))

    

class Index:
    def __init__(self,image_list:list):
        self.image_list = image_list
        if 'meta-data-files' not in os.listdir():
            os.makedirs("meta-data-files")
        self.FE = FeatureExtractor()
    def start_feature_extraction(self):
        image_data = pd.DataFrame()
        image_data['images_paths'] = self.image_list
        f_data = self.FE.get_feature(self.image_list)
        image_data['features']  = f_data
        image_data = image_data.dropna().reset_index(drop=True)
        image_data.to_pickle(config.image_data_with_features_pkl)

        print("Image Meta Information Saved: [meta-data-files/image_data_features.pkl]")
        return image_data
    def start_indexing(self,image_data):
        self.image_data = image_data
        f = len(image_data['features'][0])
        t = AnnoyIndex(f, 'euclidean')
        for i,v in tqdm(zip(self.image_data.index,image_data['features'])):
            t.add_item(i, v)
        t.build(100)
        print("Saved the Indexed File:"+"[meta-data-files/image_features_vectors.ann]")
        t.save(config.image_features_vectors_ann)
    def Start(self):
        if len(os.listdir("meta-data-files/"))==0:
            data = self.start_feature_extraction()
            self.start_indexing(data)
        else:
            print("Metadata and Features are allready present, Do you want Extract Again? Enter yes or no")
            flag  = str(input())
            if flag.lower() == 'yes':
                data = self.start_feature_extraction()
                self.start_indexing(data)
            else:
                print("Meta data allready Present, Please Apply Search!")
                print(os.listdir("meta-data-files/"))

class SearchImage:
    def __init__(self):
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl)
        self.f = len(self.image_data['features'][0])
    def search_by_vector(self,v,n:int):
        self.v = v 
        self.n = n  
        u = AnnoyIndex(self.f, 'euclidean')
        u.load(config.image_features_vectors_ann) 
        index_list = u.get_nns_by_vector(self.v, self.n) 
        return dict(zip(index_list,self.image_data.iloc[index_list]['images_paths'].to_list()))
    def get_query_vector(self,image_path:str):
        self.image_path = image_path
        img = Image.open(self.image_path)
        fe = FeatureExtractor()
        query_vector = fe.extract(img)
        return query_vector
    def plot_similar_images(self,image_path:str):
        self.image_path = image_path
        query_vector = self.get_query_vector(self.image_path)
        img_list = list(self.search_by_vector(query_vector,4).values())
        axes=[]
        fig=plt.figure(figsize=(20,15))
        for a in range(2*2):
            axes.append(fig.add_subplot(2, 2, a+1))  
            plt.axis('off')
            plt.imshow(Image.open(img_list[a]))
        fig.tight_layout()
        fig.suptitle('Similar Result Found', fontsize=22)
        plt.show(fig)
    def get_similar_images(self,image_path:str,number_of_images:int):
        self.image_path = image_path
        self.number_of_images = number_of_images
        query_vector = self.get_query_vector(self.image_path)
        img_dict = self.search_by_vector(query_vector,self.number_of_images)
        return img_dict
    def getPickle():
        object = pd.read_pickle(config.image_data_with_features_pkl)
        return object['features'][0]

class Encrytion:
    def __init__(self):
        self.fileSourcePath = ""
        self.resultOutputPath = ""

    def pixelateImages(img, name, resultOutputPath) :
        extension = ".jpg"
        imgSmall = img.resize((70,70),resample=Image.BILINEAR)
        result = imgSmall.resize(img.size,Image.NEAREST)
        print(resultOutputPath+name)
        result.save(resultOutputPath + name + extension)        

    def iterateAndParseFolder(self,fileSourcePath,resultOutputPath):
        print(fileSourcePath)
        images = glob(fileSourcePath)
        for image in images:
            with open(image, 'rb') as file:
                img = Image.open(file)
                filename = os.path.basename(file.name).split('.')[0]
                Encrytion.pixelateImages(img, filename,resultOutputPath)


    

# image_list = LoadData().from_folder(["D:\Project\Final Year\Friend Rcommndation System\Final 2\Images\\training"])
# Index(image_list).Start()
#
# object = pd.read_pickle(config.image_data_with_features_pkl)
# print(object)
# fin = SearchImage().get_similar_images(image_path="D:\Project\Final Year\Friend Rcommndation System\OpenCV\Query Image\IMG_20210223_213530_Bokeh.jpg",number_of_images=1)
# print(fin)
# #
# fe = FeatureExtractor()
# # fe.results()
# fe.cust_dataset()
#
# SearchImage().plot_similar_images("D:\Project\Final Year\Friend Rcommndation System\Final 2\Images\query\\00e3c8ff3453fc3224e4a01bb393db1c.jpg")

# fileSourcePath = "D:\Project\Final Year\Friend Rcommndation System\Final 2\images\\apparels\images\*.jpg"
# resultOutputPath = "D:\Project\Final Year\Friend Rcommndation System\Final 2\\target\\blurred_image\\"
# enc = Encrytion()
# enc.iterateAndParseFolder(fileSourcePath,resultOutputPath)