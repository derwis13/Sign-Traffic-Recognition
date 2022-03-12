import cv2
import keras
import numpy as np
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

photo_shape=np.array([30,30,3])


def load(filepath,filename):

    #open and read csv file
    file=open(filepath+filename,'r').read()
    lines=file.split('\n')
    #save first line as legend
    #legend's format: width,height, 4 points bndbox [x,y,x+width,y+height],class Id photo, Path to photo
    legend=lines.pop(0).split(',')
    print(legend)

    listDict=[]

    #save information about each photo to list
    for line in lines:
        item=line.split(',')
        if len(item)>1:
            dict = {}
            for i in range(len(legend)):
                dict[legend[i]]=item[i]
            listDict.append(dict)
    return listDict

def readCropImage(list,filepath):

    data = []
    #for each list item, open and save crop photo (area of the sign) to directories list.
    #format of directory: key:"image" - value:image, key:"label" - value:"ClassId"
    for i in list:
        img = cv2.imread(filepath+i['Path'])[int(i['Roi.Y1']):int(i['Roi.Y2']),int(i['Roi.X1']):int(i['Roi.X2'])]
        img = cv2.resize(img,(photo_shape[:2]))
        data.append({'image':img, 'label':i['ClassId']})

    return data

def readImage(list,filepath):
    data = []

    #for each list item, open and save crop photo (area of the sign) to directories list.
    #format of directory: key:"image" - value:image,  key:"label" - value:ClassId sign at photo
    #                     key:"bndbox"- value:bndbox of sign, key:"size"  - value:size photo [width,height]
    for i in list:
        img = cv2.imread(filepath+i['Path'])
        img = cv2.resize(img,(photo_shape[:2]))
        bndbox=[int(i['Roi.X1']),int(i['Roi.Y1']),int(i['Roi.X2']),int(i['Roi.Y2'])]
        size=[int(i['Width']),int(i['Height'])]
        data.append({'image':img, 'label':i['ClassId'], 'bndbox':bndbox, 'size':size})

    return data

def build_cnn_model(data_):

    labels=[]
    data=[]

    #save images and labels to lists
    for dict in data_:
        data.append(dict['image'])
        labels.append(dict['label'])

    data = np.array(data)
    labels=np.array(labels)
    #map labels to int format
    labels=list(map(int, labels))

    #split to train database and test database in proportion 5:1
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    labels_test=to_categorical(labels_test,43)
    labels_train=to_categorical(labels_train,43)

    #set model parameters
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=photo_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #fit model
    model.fit(data_train,labels_train,validation_data=(data_test,labels_test),batch_size=32, epochs=15)
    return model

def predict1(model,data_):
    labels = []
    data = []

    for dict in data_:
        data.append(dict['image'])
        labels.append(dict['label'])

    data = np.array(data)
    labels = np.array(labels)
    labels = list(map(int, labels))

    #predict labels from each images from list
    pred= np.argmax(model.predict(data), axis=1)

    #calculate and pritn accuracy predict
    print(accuracy_score(labels, pred))

    return pred


#load train database
dataTrain=readCropImage(load('','Train.csv'),'')

#load test database
dataTest=readCropImage(load('','Test.csv'),'')
#dataTest=readImage(load('','Train.csv'),'')

#build cnn model
model=build_cnn_model(dataTrain)

#save model
model.save_weights('my_weights.h5')
#model.save('my_model')

###########################################################
#addictional:

#load model
model=keras.models.load_model("my_model")

#check accuracy predict and print predict labels from dataTest list
predict1(model,dataTest)