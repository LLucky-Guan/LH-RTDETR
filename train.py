import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/LH-RTDETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='hat-data.yaml',
                cache=False,
                imgsz=640,
                epochs=111,
                batch=8, 
                workers=4, 
                # device='0,1', 
                # resume='', # last.pt path
                project='runs/train',
                name='LH-RTDETR'
                )