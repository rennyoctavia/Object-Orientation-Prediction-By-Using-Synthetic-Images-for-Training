import cv2
import numpy as np
import os

def get_yolo_v3_model(path='../yolo-coco/'):

    # get class labels
    with open(path + 'coco.names', 'r') as labels_file:
        labels = labels_file.read().strip().split("\n")

    # create path to yolo model weights
    weights_path = path + 'yolov3.weights'

    # create path to yolo model config
    config_path = path + 'yolov3.cfg'

    # create the yolo model
    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    return model, labels

def apply_model_to_image(model, labels, image, min_confidence=0.5, threshold=0.3):

    # get relevant output layers from the model
    ln = model.getLayerNames()
    ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    # read the image
    H, W = image.shape[:2]

    # create a blob from image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # give the blob to the model
    model.setInput(blob)

    # get bounding boxes from model
    output_layers = model.forward(ln)

    # initialize neccessary lists
    boxes = []
    confidences = []
    classIDs = []

    # iterate over output layers
    for output in output_layers:

        # iterate over detected objects
        for detection in output:

            # get class ID
            scores = detection[5:]
            classID = np.argmax(scores)

            # filter out chair
            if labels[classID] == 'chair':

                # get confidence of detected object
                confidence = scores[classID]

                # skip objects with weaker confidence then min_confidence
                if confidence > min_confidence:

                    # scale bounding box to input image
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")

                    # get top left corner coordinate
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # append to neccessary lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    # get rid of overlapping weaker boxes (to prevent re-detection of same object)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, threshold)

    # create list of cropped images
    cropped = []
    coords = []
    dims = []

    # if any object detected
    if len(idxs) > 0:

        # iterate over each object detected
        for i in idxs.flatten():

            # bounding box coordinates
            x, y = (boxes[i][0], boxes[i][1])
            width, height = (boxes[i][2], boxes[i][3])

            # get cropped image of object
            cropped_image = image[y:y+height, x:x+width]
            
            # make sure we have a valid image
            if cropped_image.size > 0:
                cropped_image = cv2.resize(cropped_image,(64,64))
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                cropped.append(cropped_image.astype(np.float))
                coords.append(np.array([x, y]))
                dims.append(np.array([width, height]))

    return cropped, coords, dims

def use_webcam(callback):
    model, labels = get_yolo_v3_model()
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if ret_val:
            cropped, coords, dims = apply_model_to_image(model, labels, img)
            callback(img, cropped, coords, dims)
        if cv2.waitKey(1) == 27: 
            cv2.destroyAllWindows()
            break

def use_image(img, callback):
    model, labels = get_yolo_v3_model(path='yolo-coco/')
    cropped, coords, dims = apply_model_to_image(model, labels, img)
    callback(img, cropped, coords, dims)

def get_cropped_image(img):
    model, labels = get_yolo_v3_model()
    cropped, coords, dims = apply_model_to_image(model, labels, img)
    return img, cropped, coords, dims
    
def test(img, cropped, coords, dims):
    cv2.imshow('Whole image', img)
    count = 1
    for c in cropped:
        cv2.imshow('Crop nr. ' + str(count) +'.', c)
        print(coords[count-1], dims[count-1])
        count += 1

#use_webcam(test)

#img = cv2.imread('octiba/object_detection/test.jpg')
#print(type(img))
#use_image(img, test)