import cv2
import numpy as np
from torch import device
from predict import load_model, preprocessing, \
    postprocessing, load_config, CONFIG_PATH

np.set_printoptions(suppress=True)

def main():
    #Load config from yaml
    config_dict = load_config(CONFIG_PATH)

    #Read img0 to BGR
    img0 = cv2.imread('inference/images/image1.jpg')
    original_shape = img0.shape 
    
    #Load model
    model, stride, device = load_model(config_dict, './yolov7.pt')

    #Resize, pad
    img = preprocessing(config_dict, img0, stride, device) 
    inference_shape = img.shape

    #Inference
    pred = model(img, augment=config_dict['augment'])[0] #Shape (1, num_preds, 85)

    #nms and scale coordinates
    pred = postprocessing(config_dict, pred, original_shape, inference_shape)
 
if __name__ == '__main__':
    main()