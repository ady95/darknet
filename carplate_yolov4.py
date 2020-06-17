from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet




class CarplateYoloV4:

    THRESHOLD = 0.5

    def __init__(self):

        # global metaMain, netMain, altNames
    
        base_path = r"E:\GITCAR\darknet\train_carplate"

        configPath = os.path.join(base_path, "yolov4-carplate.cfg")
        weightPath = os.path.join(base_path, "yolov4-carplate_final.weights")
        metaPath = os.path.join(base_path, "carplate.data")

        self.netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        self.metaMain = darknet.load_meta(metaPath.encode("ascii"))

        # try:
        #     with open(metaPath) as metaFH:
        #         metaContents = metaFH.read()
        #         import re
        #         match = re.search("names *= *(.*)$", metaContents,
        #                           re.IGNORECASE | re.MULTILINE)
        #         if match:
        #             result = match.group(1)
        #         else:
        #             result = None
        #         try:
        #             if os.path.exists(result):
        #                 with open(result) as namesFH:
        #                     namesList = namesFH.read().strip().split("\n")
        #                     self.altNames = [x.strip() for x in namesList]
        #         except TypeError:
        #             pass
        # except Exception:
        #     pass


        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain),3)
    
    @staticmethod
    def convertBack(x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    @staticmethod
    def cvDrawBoxes(detections, img):
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        return img


    def predict_file(self, image_path):
        frame_read = cv2.imread(file_path)
        return self.predict(frame_read)

    def predict(self, image_np):
        frame_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain)),
                                   interpolation=cv2.INTER_AREA)

        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=self.THRESHOLD)

        return detections, frame_resized

    def get_alpr_image(self, image_np):
        detections, frame_resized = self.predict(image_np)
        if len(detections) == 0: return None, None

        img_h, img_w, _ = image_np.shape
        rimg_h, rimg_w, _ = frame_resized.shape

        detection = detections[0]

        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = self.convertBack(
            float(x), float(y), float(w), float(h))

        ratio_x = img_w / rimg_w
        ratio_y = img_h / rimg_h

        xmin = int(xmin * ratio_x)
        xmax = int(xmax * ratio_x)
        ymin = int(ymin * ratio_y)
        ymax = int(ymax * ratio_y)

        caplate_np = image_np[ymin:ymax, xmin:xmax]

        carplate_box = {
            "x": xmin,
            "y": ymin,
            "w": xmax - xmin,
            "h": ymax - ymin,
        }

        return caplate_np, carplate_box


if __name__ == "__main__":
    import json

    yolo = CarplateYoloV4()

    BASE_FOLDER = r"E:\DATA\@car\carphoto\car_object_crop\1"
    files = os.listdir(BASE_FOLDER)

    # Create an image we reuse for each detect
    for file in files:
        prev_time = time.time()
        # ret, frame_read = cap.read()
        
        file_path = os.path.join(BASE_FOLDER, file)
        # frame_read = cv2.imread(file_path)
        # frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        
        # frame_resized = cv2.resize(frame_rgb,
        #                            (darknet.network_width(netMain),
        #                             darknet.network_height(netMain)),
        #                            interpolation=cv2.INTER_LINEAR)

        # darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        # detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)
        # detections, frame_resized = yolo.predict_file(file_path)
        # image = cvDrawBoxes(detections, frame_resized)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np = cv2.imread(file_path)
        carplate_np, carplate_box = yolo.get_alpr_image(image_np)
        if type(carplate_np) != np.ndarray: continue

        print(1/(time.time()-prev_time))

        # img_list = carplate_np.tolist()
        # img_json = json.dumps(img_list)
        # print(img_json)

        # carplate_np2 = np.array(json.loads(img_json))
        # print(carplate_np2)
        cv2.imshow('Demo', carplate_np)
        cv2.waitKey(0)
        # cv2.waitKey(3)



