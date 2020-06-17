import os
import json
import requests
import time
import base64

from beyond_car import image_util

# 번호판 보정 + OCR
carplate_url = "http://127.0.0.1:9002/predict_carplate"
# carplate_url = "http://ai.nextlab.co.kr:8088/predict_carplate"


BASE_FOLDER = r"E:\DATA\@car\carphoto\car_object_crop\1"

for filename in os.listdir(BASE_FOLDER):
    
    start_time = time.time()

    filepath = os.path.join(BASE_FOLDER, filename)
    print(filepath)
    
    b64_image = image_util.encode_base64_from_file(filepath)


    param_dict = {
        "base64_image": b64_image,
    }
    ret = requests.post(carplate_url, json=param_dict)
    result = ret.json()

    predict_time = time.time() - start_time

    print(predict_time, filename, result)




