import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

photo_shape=np.array([30,30,3])

def load(filepath,filename):
    file=open(filepath+filename,'r').read()
    lines=file.split('\n')
    legend=lines.pop(0).split(',')
    print(legend)

    listDict=[]

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
    for i in list:
        img = cv2.imread(filepath+i['Path'])[int(i['Roi.Y1']):int(i['Roi.Y2']),int(i['Roi.X1']):int(i['Roi.X2'])]
        img = cv2.resize(img,(photo_shape[:2]))
        data.append({'image':img, 'label':i['ClassId']})

    return data

def readImage(list,filepath):
    data = []
    for i in list:
        img = cv2.imread(filepath+i['Path'])
        img = cv2.resize(img,(photo_shape[:2]))
        bndbox=[int(i['Roi.Y1']),int(i['Roi.Y2']),int(i['Roi.X1']),int(i['Roi.X2'])]
        #width,height
        size=[int(i['Width']),int(i['Height'])]
        data.append({'image':img, 'label':i['ClassId'], 'bndbox':bndbox, 'size':size})

    #print(data)
    return data



def build_cnn_model(data_):

    labels=[]
    data=[]

    for dict in data_:
        data.append(dict['image'])
        labels.append(dict['label'])

    data = np.array(data)
    labels=np.array(labels)
    labels=list(map(int, labels))

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    labels_test=to_categorical(labels_test,43)
    labels_train=to_categorical(labels_train,43)


    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=photo_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    #model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu')) #add
    #model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # add
    #model.add(MaxPool2D(pool_size=(2, 2))) #add
    #model.add(Dropout(rate=0.25)) #add

    #model.add(MaxPool2D(pool_size=(2, 2))) #add
    #model.add(Dropout(rate=0.25)) #add
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(data_train,labels_train,validation_data=(data_test,labels_test),batch_size=32, epochs=3)

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

    pred= np.argmax(model.predict(data), axis=1)

    print(accuracy_score(labels, pred))

    #return pred



#dataTrain=readCropImage(load('','Train.csv'),'')
#dataTest=readCropImage(load('','Test.csv'),'')

dataTest=readImage(load('','Train.csv'),'')

#for img in dataTest
#

#model=build_cnn_model(dataTrain)
#predict1(model,dataTest)



##########################################################

def boVW(data):
    sift = cv2.SIFT_create()
    dict_size = 500#128
    bow = cv2.BOWKMeansTrainer(dict_size)
    for dict in data:
        kp, des = sift.detectAndCompute(dict['image'], None)
        if des is not None:
            bow.add(des)
    vocabulary=bow.cluster()
    return vocabulary

def extractingFeautres(vocabulary,data):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow_extr = cv2.BOWImgDescriptorExtractor(sift,flann)
    bow_extr.setVocabulary(vocabulary)

    data_feautres=[]

    for dict in data:
        histogram=[]
        key,des=sift.detectAndCompute(dict['image'],None)
        if des is not None:
            histogram=bow_extr.compute(dict['image'],key)
        data_feautres.append({'data': histogram, 'label': dict['label']})

    return data_feautres

def train(data):
    X=[]
    y=[]
    for dict in data:
        try:
            X.append(dict['data'][0])
            y.append(dict['label'])
        except:
            X.append(np.zeros(500))
            y.append('other')

    clf=RandomForestClassifier(n_estimators=1000)
    clf.fit(X,y)
    return clf

def predict(clf,data_test):
    labels=[]
    X_data=[]
    for dict in data_test:
        try:
            X_data.append(dict['data'][0])
            labels.append(dict['label'])
        except:
            X_data.append(np.zeros(500))
            labels.append('other')

    pred=clf.predict(X_data)
    print(accuracy_score(labels,pred))


