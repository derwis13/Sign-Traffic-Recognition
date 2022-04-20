import tensorflow as tf
import cv2
from tflite_model_maker import object_detector
from tflite_model_maker import model_spec
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import tflite_support.metadata_writers
from tflite_support import metadata

MODEL_FILE='model_1.tflite'
PHOTO='image_1.jpg'


def detectOnImage(image,boxes,classes,scores,dict_classes,req_scores=0.6):
    height,width,_=image.shape

    for i in range(len(boxes)):
        color_bndbox=(0,0,255)
        if scores[i]>=req_scores:
            name_class=dict_classes.get(classes[i])
            image=cv2.rectangle(image,
                                (int(boxes[i][1]*width),int(boxes[i][0]*height)),
                                (int(boxes[i][3]*width),int(boxes[i][2]*height)),
                                color_bndbox)
            (w, h), _ = cv2.getTextSize(name_class,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        1)
            imgage = cv2.putText(image,
                                 name_class,
                                 (int(boxes[i][1]*width),int(boxes[i][0]*height)-5),
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 0.6,
                                 color_bndbox,
                                 1)
    return image

displayer = metadata.MetadataDisplayer.with_model_file(MODEL_FILE)
s=displayer.get_associated_file_buffer('labelmap.txt').decode('utf-8')
a=-1
dict_classes={}
for i in s.split('\n'):
    dict_classes.update({a:i})
    a+=1
print(dict_classes)

#optional save labels as labels.txt

#with open('labels.txt', 'w') as f:
    #f.write(str(s))

# Load Image
img=cv2.imread(PHOTO)

# Resize Image
img=cv2.resize(img,(300,300))
img=img.reshape([1, 300, 300, 3])

# Load TFLite model and allocate tensors.
interpreter=tf.lite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img)
interpreter.invoke()


# Get tensor from Image
bndboxes=interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]
classes=interpreter.get_tensor(interpreter.get_output_details()[3]['index'])[0]
scores=interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
num=interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]

#old version:
bndboxes=interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
classes=interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]
scores=interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]
num=interpreter.get_tensor(interpreter.get_output_details()[3]['index'])[0]

print('box:',bndboxes,'\nclasses:',classes,'\nscores:',scores,'\nnum:',num)

img=img.reshape([300,300,3])


# Detect objects on Image and draw rectangle
img=detectOnImage(img,bndboxes,classes,scores,dict_classes,0.5)

# Show image
cv2.imshow('image', img)
cv2.waitKey(0)

# Optional - save image
#cv2.imwrite('salatka.jpg', img)