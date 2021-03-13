import sys
import argparse
from yolo import YOLO, detect_video,detect_video_train, load_data,save_pics
from PIL import Image

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    #save_pics(YOLO())
    #detect_video_train(YOLO())
    detect_video(YOLO())
    #load_data()
    #else:
    #    print("Must specify at least video_input_path.  See usage with --help.")
