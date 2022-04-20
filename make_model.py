import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from PIL import Image
import glob
import xml.etree.ElementTree as ET
import csv

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

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

def load_xml():
    fileformat='xml'
    xml_files=[]

    for each_file in glob.glob('annotations\\*.{}'.format(fileformat)):
        xml_files.append(each_file)
    data=[]
    for each in xml_files:


        size_image = []
        list_objects_on_image =[]
        tree=ET.ElementTree(file=each)
        root=tree.getroot()
        file_name=root[1].text
        folder=root[0].text
        path=folder+'/'+file_name
        size_image.append({'width':root[2][0].text,'height':root[2][1].text,'depth':root[2][2].text})
        for child in root.iter('object'):
            bndbox = []
            for bndbox_ in child.iter('bndbox'):
                bndbox.append([bndbox_[0].text,bndbox_[1].text,bndbox_[2].text,bndbox_[3].text])
            list_objects_on_image.append({'object':child[0].text,'bndbox':bndbox[0]})

        data.append({'path':path,'size':size_image[0],'objects':list_objects_on_image})
    return data

def load_txt(filepath,filename):
    file = open(filepath + filename, 'r').read()
    lines = file.split('\n')
    data=[]
    for each in lines:
        bndbox = []
        size_image=[]
        s=each.split(';')

        im = Image.open(filepath+s[0])
        im.save(filepath+s[0].replace('ppm','png'))
        size_image.append({'width':int(im.width),'height':int(im.height)})

        bndbox.append({'bndbox':[int(s[1]),int(s[2]),int(s[3]),int(s[4])],'object':s[5]})
        data.append({'path':s[0].replace('ppm','png'),'size':size_image[0],'objects':bndbox})
    return data

def create_csv(data,path):
    with open(path, 'w', newline='') as csvfile:
        writer=csv.writer(csvfile,delimiter=',')
        for data_ in data:
            for objects in data_['objects']:
                print(data_['size']['width'],data_['size']['height'],objects['bndbox'][0],
                      objects['bndbox'][1],objects['bndbox'][2],objects['bndbox'][3],
                      objects['object'],data_['path'])
                writer.writerow([data_['size']['width'],data_['size']['height'],objects['bndbox'][0],
                      objects['bndbox'][1],objects['bndbox'][2],objects['bndbox'][3],
                      objects['object'],data_['path']])
def reformat_csv(oldPath,newPath,set):
    data=[]
    with open(oldPath,newline='') as csvfile:
        reader=csv.reader(csvfile,delimiter=';')
        for row in reader:
            temp=row[0].split(',')
            data.append({'width':temp[0],'height':temp[1],'x1':temp[2],'y1':temp[3],'x2':temp[4],'y2':temp[5],'classid':temp[6],'path':temp[7]})
    print(data.pop(0))
    with open(newPath, 'w', newline='') as csvfile:
        writer=csv.writer(csvfile,delimiter=',')
        for data_ in data:
            writer.writerow([set,data_['path'].replace('png','jpg'),data_['classid'],
                             round(int(data_['x1'])/int(data_['width']),4),round(int(data_['y1'])/int(data_['height']),4),'','',
                             round(int(data_['x2'])/int(data_['width']),4),round(int(data_['y2'])/int(data_['height']),4),'',''])
def convertImage(directoryPath):
    data=load('',directoryPath)
    for i in data:
        #print(i)
        save_path = 'C:/Users/Kuba/Desktop/FullIJCNN2013/'+i['Path'].replace('png', 'jpg')
        try:
            image=Image.open('C:/Users/Kuba/Desktop/FullIJCNN2013/'+i['Path'])
            image1 = image.convert('RGB')
            os.remove('C:/Users/Kuba/Desktop/FullIJCNN2013/'+i['Path'])
            image1.save(save_path)
        except:pass

def compare_csv(trainDirPath,testDirPath,compareDirPath):
    reader=[]
    with open(trainDirPath,newline='') as csvfile:
        for row in csv.reader(csvfile,delimiter=' '):
            reader.append(row)
    with open(testDirPath, newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter=' '):
            reader.append(row)
    with open(compareDirPath, 'w', newline='') as csvfile:
        writer=csv.writer(csvfile,delimiter=' ')
        for i in reader:
            temp=i[0]
            writer.writerow([temp])


PATH_to_directory='C:/Users/Kuba/Desktop/FullIJCNN2013/Train1.csv'
PATH_to_images='C:/Users/Kuba/Desktop/FullIJCNN2013'

#prepare dataset:

#data=load_txt(PATH_to_images+'/','gt.txt')
#create_csv(data,PATH_to_directory)
#convertImage(PATH_to_directory)
#reformat_csv(PATH_to_directory,PATH_to_directory,'TRAIN')


#create model:
data=object_detector.DataLoader.from_csv(PATH_to_directory,images_dir=PATH_to_images)
model=object_detector.create(data[0],model_spec=model_spec.get('efficientdet_lite0'),batch_size=8,epochs=20)
model.export(export_dir='.')
