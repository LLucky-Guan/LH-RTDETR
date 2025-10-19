import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info

if __name__ == '__main__':
    model_path = 'runs/train/LH-RTDETR/weights/best.pt'
    model = RTDETR(model_path) # 选择训练好的权重路径
    result = model.val(data='data.yaml',
                      split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
                      imgsz=640,
                      batch=6,
                      save_json=False, # if you need to cal coco metrice
                      project='runs/val',
                      name='LH-RTDETR',
                      )
