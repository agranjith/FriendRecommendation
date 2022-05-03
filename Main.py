from pyexpat import model
from re import T
from select import select
from tabnanny import verbose
import DeepImageSearch.config as config
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import layers,models
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img

import scipy
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

datasetdir = "D:\Project\Final Year\Friend Rcommndation System\FInal\\fi\Images"
os.chdir(datasetdir)

class Accuracy:

    def generators(shape, preprocessing):

        batch_size = 30 

        imgdatagen = ImageDataGenerator(
            preprocessing_function = preprocessing,
            horizontal_flip = True, 
            validation_split = 0.1,
        )

        height, width = shape

        train_dataset = imgdatagen.flow_from_directory(
            os.getcwd(),
            target_size = (height, width), 
            classes = ('query','training'),
            batch_size = batch_size,
            subset = 'training', 
        )

        val_dataset = imgdatagen.flow_from_directory(
            os.getcwd(),
            target_size = (height, width), 
            classes = ('query','training'),
            batch_size = batch_size,
            subset = 'validation'
        )
        return train_dataset, val_dataset

    def plot_history(history, yrange):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc)
        plt.plot(epochs, val_acc)
        plt.title('Training and validation accuracy')
        plt.ylim(yrange)

        plt.figure()

        plt.plot(epochs, loss)
        plt.plot(epochs, val_loss)
        plt.title('Training and validation loss')

        plt.show()
    


    

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
    def acc(self):
        vgg16 = keras.applications.vgg16
        train_dataset, val_dataset = Accuracy.generators((224,224), preprocessing=vgg16.preprocess_input)
        conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
        x = keras.layers.Flatten()(conv_model.output)
        x = keras.layers.Dense(100, activation='relu')(x)
        x = keras.layers.Dense(100, activation='relu')(x)
        x = keras.layers.Dense(100, activation='relu')(x)

        predictions = keras.layers.Dense(2, activation='softmax')(x)

        full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
        full_model.summary()
        for layer in conv_model.layers:
            layer.trainable = False
        full_model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                  metrics=['acc'])
        history = full_model.fit(
            train_dataset, 
            validation_data = val_dataset,
            workers=10,
            epochs=5,
        )
        Accuracy.plot_history(history, yrange=(0.9,1))
        
    

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
        print(image_data.shape)
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
        img_list = list(self.search_by_vector(query_vector,16).values())
        print(img_list)
        axes=[]
        fig=plt.figure(figsize=(20,15))
        for a in range(4*4):
            axes.append(fig.add_subplot(4, 4, a+1))  
            plt.axis('off')
            plt.imshow(Image.open(img_list[a]))
        fig.tight_layout()
        fig.suptitle('Similar Result Found', fontsize=22)
        plt.show(fig)
        plt.plot()
    def get_similar_images(self,image_path:str,number_of_images:int):
        self.image_path = image_path
        self.number_of_images = number_of_images
        query_vector = self.get_query_vector(self.image_path)
        img_dict = self.search_by_vector(query_vector,self.number_of_images)
        return img_dict
    def getPickle():
        object = pd.read_pickle(config.image_data_with_features_pkl)
        return object['features'][0]

    

# image_list = LoadData().from_folder(["D:\Project\Final Year\Friend Rcommndation System\OpenCV\Training Set"])
# Index(image_list).Start()

# object = pd.read_pickle(config.image_data_with_features_pkl)
# print(object)
# fin = SearchImage().get_similar_images(image_path="D:\Project\Final Year\Friend Rcommndation System\OpenCV\Query Image\IMG_20210223_213530_Bokeh.jpg",number_of_images=1)
# print(fin)

# fe = FeatureExtractor()
# fe.acc()

# SearchImage().plot_similar_images("D:\Project\Final Year\Friend Rcommndation System\OpenCV\Query Image\IMG_20210228_133621_Bokeh.jpg")