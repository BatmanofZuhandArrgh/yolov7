import os
import matplotlib
import yaml
import numpy as np
from pprint import pprint

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

CONFIG_PATH = './predict_config.yaml'
COCO_CONFIG = './data/coco.yaml'

def load_model(config_dict, weight_path = None):
    # Load model based on config yaml and return model in RAM
    
    # Initialize
    set_logging()
    device = select_device(config_dict['device'])
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # weight_path parameters overwrites the path in config dict if is not None
    model_path = weight_path if weight_path is not None else config_dict['weights']
    model = attempt_load(model_path, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(config_dict['img_size'], s=stride)  # check img_size

    #Return traced jit torch model
    #"Trace a function and return an executable or ScriptFunction that will be optimized using just-in-time compilation." 
    if not config_dict['no_trace']:
        model = TracedModel(model, device, config_dict['img_size'])
    
    # For lighter model and faster inference
    if half:
        model.half()  # to FP16

    # Second-stage classifier 
    # # (Apparently in the og repo, no off-the-shell classifier weights are provided)
    # if config_dict['classify']:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    # else:
    #     modelc = None

    return model, stride, device #, modelc

#From utils/datasets.py
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocessing(config_dict, frame, stride, device):
    # Device config
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # In datasets.py class LoadImages
    img = letterbox(frame, config_dict['img_size'], stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Processing in detect.py
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img

def postprocessing(config_dict, pred, original_shape, inference_shape):
    '''
    Output: numpy array in shape (num_prediction, 6)
    [min_x, min_y, max_x, max_y, conf_score] * num_preds  
    '''
    # Apply NMS
    pred = non_max_suppression(pred, config_dict['conf_thres'], config_dict['iou_thres'], classes=config_dict['classes'], agnostic=config_dict['agnostic_nms'])
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(inference_shape[2:], det[:, :4], original_shape).round()
    
    return pred[0].detach().numpy()

def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config_dict

#From utils/plot.py
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)

def plot_image(preds, img, save_path='images.jpg', class_names = None):
    for pred in preds:
        cls = int(pred[-1])
        conf = float(pred[-2])

        colors = color_list()  # list of colors
        color = colors[cls % len(colors)]
        cls = class_names[cls] if class_names else cls

        label = '%s %.1f' % (cls, conf)
        plot_one_box(pred, img, color=color, label = label)
  
        
    cv2.imwrite(save_path, img) 
    

def main():
    config_dict = load_config(CONFIG_PATH)
    pprint(config_dict) 
    coco_dict = load_config(COCO_CONFIG)
    class_names = coco_dict['names']
    
    source_path = config_dict['source']
    #Read img0 to BGR
    img0 = cv2.imread(source_path)
    original_shape = img0.shape 
    
    #Load model
    model, stride, device = load_model(config_dict, './yolov7.pt')

    #Resize, pad
    img = preprocessing(config_dict, img0, stride, device) 
    inference_shape = img.shape

    #Inference
    preds = model(img, augment=config_dict['augment'])[0] #Shape (1, num_preds, 85)

    #nms and scale coordinates
    preds = postprocessing(config_dict, preds, original_shape, inference_shape)
    
    print(preds)
    if len(preds) != 0 and config_dict['no_save'] == False:
        fname = os.path.basename(source_path)
        plot_image(preds, img0, save_path = os.path.join(config_dict['project'], fname),
                   class_names = class_names,
                   )

if __name__ == '__main__':
    main()

