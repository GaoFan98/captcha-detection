import cv2
import numpy as np
import os
import argparse
import base64

parser = argparse.ArgumentParser()
from PIL import Image
from io import BytesIO

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--base64', help='base64 string.')
args = parser.parse_args()

# Captcha Model Files
number_weights_path = "./model/recognition.weights"
number_cfg_path = "./model/recognition.cfg"
number_classes_path = "./model/recognition.txt"

# load the captcha class labels our YOLO model was trained on
number_net = cv2.dnn.readNet(number_weights_path, number_cfg_path)
number_classes = None
with open(number_classes_path, 'r') as f:
    number_classes = [line.strip() for line in f.readlines()]
# determine only the *output* layer names that we need from YOLO
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def process_plate(image):
    scale = 0.00392

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities


    blob = cv2.dnn.blobFromImage(image, scale, (200, 100), (0, 0, 0), True, crop=False)
    number_net.setInput(blob)
    # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
    outs = number_net.forward(get_output_layers(number_net))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively

    class_ids = []
    confidences = []
    boxes = []
    center_X = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    # loop over each of the layer outputs
    for out in outs:
        # loop over each of the detections
        for detection in out:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, w, h])
                center_X.append(center_x)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    result = ''
    valid_boxes = []
    valid_classids = []
    valid_centerX = []

    # Sorting of box Infos
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        valid_boxes.append(box)
        valid_classids.append(class_ids[i])
        valid_centerX.append(x)
    for i in range(0, len(valid_centerX)):
        for j in range(i + 1, len(valid_centerX)):
            if valid_centerX[i] > valid_centerX[j]:
                temp = valid_centerX[i]
                valid_centerX[i] = valid_centerX[j]
                valid_centerX[j] = temp
                tem = valid_classids[i]
                valid_classids[i] = valid_classids[j]
                valid_classids[j] = tem
    for i in range(0, len(valid_classids)):
        result += number_classes[valid_classids[i]]
    return result
if __name__=='__main__':
    if(args.image is None and args.base64 is None): # Directory Reading Part
        yourpath = "./cap_dataset/"
        for root, dirs, files in os.walk(yourpath, topdown=False): # File image Reading part from Directly
            for name in files:
                image = cv2.imread(yourpath + name)
                capt_text = process_plate(image) # CaptCha Processing
                cv2.imshow("Capture Result", image) # Mat Image Displaying
                print(capt_text) # CaptCha Output
                cv2.waitKey(0) # Delay
    elif(args.image is not None):
        image = cv2.imread(args.image)  # File image Reading part
        capt_text = process_plate(image) # CaptCha Processing
        cv2.imshow("Capture Result", image) # Mat Image Displaying
        print(capt_text) # CaptCha Output
        cv2.waitKey(0) # Delay
    else:
        base64_string = args.base64
        base64_string = base64_string.replace("data:image/png;base64,", "") # Base64 Reading Part
        im = Image.open(BytesIO(base64.b64decode(base64_string))) # Converting Base64 string to Pillow Image
        open_cv_image = np.array(im) # Converting  Pillow Image to BGRA Mat Image
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGRA2BGR) # Converting BGRA Mat Image to BGR Mat Image
        capt_text = process_plate(open_cv_image) # CaptCha Processing
        cv2.imshow("Capture Result", open_cv_image) # Mat Image Displaying
        print(capt_text) # CaptCha Output
        cv2.waitKey(0) # Delay



