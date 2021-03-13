# -*- coding: utf-8 -*-

import colorsys
import os
import cv2
from timeit import default_timer as timer
from skimage.morphology import skeletonize
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model,to_categorical
from keras.layers import Conv2D,GlobalAveragePooling2D,Dropout,Dense,Flatten,BatchNormalization
from keras.optimizers import Adam
from keras import Model
#from Motor_Control import Motorize
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self):#, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        #self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def detect_person(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        pre = image_data
        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        print("Pre model feed")

        cv2.imshow("IMAGE PRE",pre)
        cv2.waitKey(1)
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        cc=0
        existing=False
        persons_detected=[]
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            if predicted_class=="person":
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                cc+=1
                persons_detected.append([left,top,right,bottom])
        if cc>0:
            existing=True
        return existing,persons_detected

    def close_session(self):
        self.sess.close()

def save_pics(yolo):
    import cv2
    targets_available=["Non_Sign","Ped_Sign","Pol_Sign"]
    id_selected=0
    for i in range(3):
        id_selected = i
        target =targets_available[id_selected]
        for (root, dirs, files) in os.walk('Img/'+str(target)):
            if files:
                for f in files:
                    print(f)
                    path = os.path.join(root, f)
                    frame = cv2.imread(path)
                    image = Image.fromarray(frame)
                    same_frame = np.copy(image)
                    # image = yolo.detect_image(image)
                    same_frame = cv2.blur(same_frame,(5,5))
                    imgray = cv2.cvtColor(same_frame, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.adaptiveThreshold(imgray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                    sk = skeletonize(thresh)
                    sk = np.asarray(sk, dtype=np.uint8)
                    sk = sk * 255
                    #cv2.imshow('skel', sk)
                    someone, roi_coords = yolo.detect_person(image)
                    if someone:
                        roi_coords = np.asarray(roi_coords)
                        for t in range(roi_coords.shape[0]):
                            left, top, right, bottom = roi_coords[t]
                            roi_sk = sk[top:bottom, left:right]
                            print(roi_coords[t])
                            #cv2.imshow("test" + str(t), roi_sk)
                            cv2.imwrite('Img/' + str(target)+'/res/'+f,roi_sk)
                            #cv2.waitKey(1)


def load_data():
    import cv2
    targets_available = ["Non_Sign", "Ped_Sign", "Pol_Sign"]
    data=np.uint8
    temp_label=[]
    temp_data=[]
    Y=[]
    otro = []
    images = np.array([], dtype=object)
    c=0
    for id_selected in range(3):
        temp_label = to_categorical(id_selected,3)
        #print("Temp_label"+str(temp_label))
        target = targets_available[id_selected]
        for (root, dirs, files) in os.walk('Img/' + str(target)+'/res/'):
            if files:
                for f in files:
                    #print(f)
                    path = os.path.join(root, f)
                    print(path)
                    print(temp_label)
                    frame = cv2.imread(path,0)
                    print(np.asarray(frame).shape)
                    frame = cv2.resize(frame,(44,146))
                    print(np.asarray(frame).shape)
                    frame = np.asarray(frame)
                    #cv2.imshow("fr",frame)
                    #cv2.waitKey(1)
                    #print(frame.shape)
                    #frame = np.reshape(frame,(-1,frame.shape[0],frame.shape[1],1))
                    frame = frame/255
                    print(frame)
                    frame = np.reshape(frame,(frame.shape[0],frame.shape[1],1))
                    temp_label = np.reshape(temp_label,(3))
                    temp_data.append(frame)
                    c+=1
                    Y.append(temp_label)



    return  temp_data,Y,c

def build_cnn():
    in_layer = Input(shape=(146, 44, 1))
    x = BatchNormalization()(in_layer)
    x = Conv2D(128, (4, 4), activation='elu')(x)  # single stride 4x4 filter for 16 maps
    x = Conv2D(64, (4, 4), activation='elu')(x)  # single stride 4x4 filter for 32 maps
    x = Dropout(0.5)(x)
    x = Conv2D(64, (4, 4), activation='elu')(x)  # single stride 4x4 filter for 64 maps
    x = Dropout(0.5)(x)
    x = Conv2D(128, (1, 1))(x)  # finally 128 maps for global average-pool
    x = Flatten()(x) # pseudo-dense 128 layer
    output_layer = Dense(3, activation="softmax")(x)  # softmax output
    model = Model(inputs=in_layer, outputs=output_layer)
    learning_rate = 1e-3
    optm = Adam(lr=learning_rate)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    #model.load_weights("alex_weights.h5")
    return model

def detect_video_save_pics(yolo):
    save_pics(yolo)

def detect_video_train(yolo):
    model=build_cnn()
    data,Y,ra =load_data()
    print("training..")
    factor = 1. / np.sqrt(2)

    model_checkpoint = ModelCheckpoint("final_alex.h5", verbose=1,
                                       monitor='val_acc', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=100, mode='max',
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    model.fit(x=np.array(data),y=np.array(Y),verbose=2,epochs=100,callbacks=callback_list,validation_split=0.15)

    print("done")

def detect_video(yolo):
    import cv2
    #print("preparing to build... cnn")
    #model = build_cnn()
    #print ("CNN done")
    print("video capture pre log")
    vid = cv2.VideoCapture(0)
    print("video capture post log")
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    targets_available = ["None Signed", "Pedestrian Sign", "Police Sign"]
    print("Entering... loop")
    while True:
        print("call_1")
        return_value, frame = vid.read()
        print("call_2")
        image = Image.fromarray(frame)
        print("call_3")
        same_frame = np.copy(image)
        print("call_4")
        result = np.asarray(image)
        print("call_5")
        same_frame = cv2.blur(same_frame,(5,5))
        imgray = cv2.cvtColor(same_frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(imgray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        sk = skeletonize(thresh)
        sk = np.asarray(sk, dtype=np.uint8)
        sk = sk*255#sk[sk == 1] = 255
        #cv2.imshow('skel', sk)
        someone,roi_coords = yolo.detect_person(image)
        Sign_Boolean =False
        if someone:
            roi_coords = np.asarray(roi_coords)
            for t in range(roi_coords.shape[0]):
                left, top, right, bottom = roi_coords[t]
                roi_sk = sk[top:bottom , left:right]
                print(roi_coords[t])
                cv2.imshow("Person_"+str(t),roi_sk)
                feed_net = cv2.resize(roi_sk, (44, 146))
                feed_net = np.asarray(feed_net)
                feed_net = feed_net /255
                feed_net = np.reshape(feed_net, (-1,feed_net.shape[0], feed_net.shape[1], 1))

                prediction = 0#model.predict(feed_net)[0]

                max_pred = 0#np.argmax(prediction)

                # cv2.imshow("fr",frame)
                # cv2.waitKey(1)
                # print(frame.shape)
                # frame = np.reshape(frame,(-1,frame.shape[0],frame.shape[1],1))
                Show_text = targets_available[max_pred]
                print("PREDICTED : "+str(Show_text)+" - "+str(prediction))
                cv2.putText(result, text=Show_text, org=(3, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2.50, color=(255, 0, 0), thickness=2)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("camera_view", cv2.WINDOW_NORMAL)
        cv2.imshow("camera_view", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

