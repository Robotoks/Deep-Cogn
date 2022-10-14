

import numpy as np
import tensorflow as tf
import cv2

def predict(interpreter, input_tensor):

  preds = interpreter(input_tensor)

  boxes = preds['detection_boxes'].numpy()
  classes = preds['detection_classes'].numpy().astype(int)
  scores = preds['detection_scores'].numpy()
  kpts = preds['detection_keypoints'].numpy()
  
  kpts_scores = preds['detection_keypoint_scores'].numpy()
  num_detections = preds['num_detections'].numpy()

  return boxes, classes, scores, num_detections, kpts, kpts_scores
  

def preprocess(img, input_size):
    WIDTH, HEIGHT = input_size
    h, w = img.shape[:2]
    if w > h:
        pad = w - h
        img_by = w
    else:
        pad = h - w
        img_by = h
    top, bottom, left, right = 0, img_by, 0, img_by
    if w > h:
        top = int(np.ceil(pad/2)) + 1
        if pad > 1:
            bottom = img_by - int(np.floor(pad/2)) + 1
        else:
            top -= 1
    else:
        left = int(np.ceil(pad/2)) + 1
        if pad > 1:
            right = img_by - int(np.floor(pad/2)) + 1
        else:
            left -= 1
    new_img = np.zeros((img_by, img_by, 3)).astype('uint8')
    new_img[top:bottom, left:right, :] = img
    input_tensor = cv2.resize(new_img, (WIDTH, HEIGHT))
    resize_ratio = min((WIDTH/w), (HEIGHT/h))
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    input_tensor = input_tensor.numpy()
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    # new_img = tf.image.resize(new_img, input_size)
    return input_tensor, new_img, (left, top)

def get_prediction(input_tensor, interpreter, from_class=1):
    (boxes, classes, scores, num_detections, kpts, kpts_scores) = predict(interpreter, input_tensor)
    kpts_sums = kpts_scores.sum(axis=2)
    kpts_sums[:, np.argwhere(classes != from_class)[:, 1]] = 0
    top_detection = np.argmax(kpts_sums)
    kpts = kpts[0, [top_detection], :, :]
    scores = scores[0, [top_detection]]
    classes = classes[0, [top_detection]]
    boxes = boxes[0, [top_detection], :]
    boxes[:,:] = 0
    kpts_scores = kpts_scores[0, [top_detection], :]

    return kpts, scores, classes, boxes, kpts_scores
