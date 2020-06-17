import sys
sys.path.append('../')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from flask import Flask, request, jsonify

import time
import numpy as np
import cv2

from beyond_car import image_util
from carplate_yolov4 import CarplateYoloV4

yolov4 = CarplateYoloV4()


app = Flask(__name__)


@app.route("/")
def home():                           
    return "Hello CarplateYoloV4 Flask"


@app.route("/predict_carplate", methods=['POST'])
def predict_carplate():
    param_dict = request.json

    base64_image = param_dict.get("base64_image")
    
    image_np = image_util.decode_base64_np(base64_image)

    carplate_np, carplate_box = yolov4.get_alpr_image(image_np)
    
    if type(carplate_np) == np.ndarray:
        carplate_base64 = image_util.encode_base64_np(carplate_np)
    else:
        carplate_base64 = None
    
    result_dict = {
        "carplate_base64": carplate_base64,
        "carplate_box": carplate_box
    }

    # 최종 이미지 검증위해 임시 저장
    # cv2.imwrite(os.path.join(r"F:\TEMP\color", param_dict.get("filename")), ret_color[2])

    return jsonify(result_dict)
    

if __name__ == "__main__":              
    app.run(host='0.0.0.0', port=9002, debug=True, use_reloader=False, threaded=False)