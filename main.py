import numpy as np
import cv2
import tensorflow as tf
from utils.configurations import KEYPOINT_DICT as kp_names
from utils.predictor import preprocess, get_prediction
from utils.drawing import draw_kps
from utils.math import Distance, ManhattanDistance
from argparse import ArgumentParser
from time import time

args = ArgumentParser()
args.add_argument('-v', '--visualize', action='store_true')
args.add_argument('-log', '--log', action='store_true')
args = args.parse_args()

videofile = str(input('Enter video path: '))
adult = str(input('Is the adult video? Enter [y]/n: ')).lower() in 'yes'
trial = str(input('Enter the trial no: '))

trials = {
    '1': {
        'seq_array_test': [
            'ear',
            'shoulder',
            'hip',
            'knee'
        ],
        'adult_time': [
            [9, 13],
            [14, 18],
            [20, 24],
            [25, 29]
        ],
        'kids_time': [
            [6, 10],
            [12, 16],
            [17, 21],
            [23, 27]
        ],
        'total_time': 31 if adult else 33
    },
    '2': {
        'seq_array_test': [
            'knee', 'ear', 'ear', 'knee', 'ear', 'knee', 'knee', 'ear'
        ],
        'adult_time': [
            [7, 11],
            [12, 16],
            [18, 22],
            [23, 27],
            [29, 33],
            [34, 38],
            [40, 44],
            [45, 49]
        ],
        'kids_time': [
          [3, 7],
          [8, 12],
          [14, 18],
          [20, 24],
          [25, 29],
          [31, 35],
          [36, 40],
          [41, 45]
        ],
        'total_time': 53 if adult else 46
    },
    '3': {
        'seq_array_test': [
            'hip', 'shoulder', 'shoulder', 'hip', 'shoulder', 'hip', 'hip', 'shoulder'
        ],
        'adult_time': [
            [7, 11],
            [12, 16],
            [18, 22],
            [23, 27],
            [29, 33],
            [34, 38],
            [40, 44],
            [45, 49]
        ],
        'kids_time': [
          [4, 8],
          [10, 14],
          [16, 20],
          [21, 25],
          [27, 31],
          [32, 36],
          [38, 42],
          [43, 47]
        ],
        'total_time': 52 if adult else 47
    },
    '4': {
        'seq_array_test': [
            'knee', 'shoulder', 'ear', 'hip', 'ear', 'shoulder', 'knee', 'ear',
            'shoulder', 'hip', 'knee', 'hip'
        ],
        'adult_time': [
            [8, 12],
            [14, 18],
            [19, 23],
            [25, 29],
            [30, 34],
            [36, 40],
            [41, 45],
            [47, 51],
            [53, 57],
            [58, 62],
            [64, 68],
            [69, 73]
        ],
        'kids_time': [
            [8, 12],
            [14, 18],
            [19, 23],
            [25, 29],
            [30, 34],
            [36, 40],
            [41, 45],
            [47, 51],
            [53, 57],
            [58, 62],
            [64, 68],
            [69, 73]
        ],
        'total_time': 75 if adult else 74
    }
}

seq_array_test = trials[trial]['seq_array_test']
adult_time = trials[trial]['adult_time']
kids_time = trials[trial]['kids_time']

# Initialize the TFLite interpreter
interpreter = None


def build_interpreter(path):
    global interpreter
    interpreter = tf.keras.models.load_model(path)
    interpreter = interpreter.signatures["serving_default"]


def create_closer(seq):
    closer_arr = dict()
    for s in seq:
        closer_arr.update({f'left_{s}': 0, f'right_{s}': 0})
    return closer_arr


def scale_up(origin_vec, scale_vec):
    d = scale_vec - origin_vec
    ext = (d + d/2.5) + origin_vec
    return ext


def find_crrect_body_parts(touched_direc):
    et = [1]
    i = 1
    while i < len(touched_direc):
        if touched_direc[i] == touched_direc[i - 1]:
            et.extend([0, 1])
            i += 2
        else:
            et.append(1)
            i += 1
    et = et[:len(touched_direc)]
    return et


st_time = time()

score_1 = 0
score_2 = 0
score_3 = 0

score_1_seq_array = seq_array_test.copy()
total_time = trials[trial]['total_time']
cap = cv2.VideoCapture(videofile)
total_frames = 0
while True:
    ok, _ = cap.read()
    if not ok:
        break
    total_frames += 1
fps = int(round(total_frames/total_time))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# cap.set(cv2.CAP_PROP_POS_FRAMES, fps * 3)

build_interpreter(path='model')
cap = cv2.VideoCapture(videofile)

threshold = 0
start_frame = fps * 0
confidence_threshold = 0.4
frame_count = 0
skip_frames = 1

width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
input_size = (512, 512)

if not cap.isOpened():
    raise IOError("video read fail")

counter = None
detection_directions = list()
touche_time = list()
empty_frames = 0
detection_frame_counts = 0

hand = None
current_ind = 0
ear_threshold = 30
hip_threshold = 30

consider_time = np.array(adult_time if adult else kids_time)
time_scores_counted = np.zeros(len(consider_time))

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        break
    if not ((frame_count % skip_frames == 0) and (frame_count/fps > consider_time[0, 0]) and (frame_count > start_frame)):
        continue
    input_tensor, frame, (left, top) = preprocess(frame[...,::-1], input_size=input_size)
    fw, fh, _ = frame.shape
    kpts, scores, classes, boxes, kpts_scores = get_prediction(input_tensor, interpreter, from_class=1)
    kpts = kpts[...,::-1]
    kpts = kpts * np.array([fw, fh])
    preds = np.concatenate([kpts, kpts_scores.reshape(1, 17, 1)], axis=2).squeeze(0)
    left_hand = preds[kp_names['left_wrist']][:2]
    right_hand = preds[kp_names['right_wrist']][:2]
    left_elbow = preds[kp_names['left_elbow']][:2]
    right_elbow = preds[kp_names['right_elbow']][:2]
    left_hand = scale_up(scale_vec=left_hand, origin_vec=left_elbow)
    right_hand = scale_up(scale_vec=right_hand, origin_vec=right_elbow)
    preds[kp_names['left_wrist'], :2] = left_hand
    preds[kp_names['right_wrist'], :2] = right_hand

    if args.visualize:

        frame = draw_kps(frame, preds)
        if top > 0:
            frame = frame[top-1:-(top-1)]
        if left > 0:
            frame = frame[:, left-1:-(left-1)]

        cv2.imshow('Frame', frame[...,::-1])
        cv2.waitKey(1)

    preds = dict(zip(kp_names, preds))

    # Thresholds
    if preds['left_ear'][2] >= 0.4 and preds['right_ear'][2] >= 0.4:
        ear_threshold = Distance(preds['left_ear'][:2], preds['right_ear'][:2])
    if preds['left_hip'][2] >= 0.4 and preds['right_hip'][2] >= 0.4:
        hip_threshold = Distance(preds['left_hip'][:2], preds['right_hip'][:2])

    if detection_frame_counts != 0:
        detection_frame_counts += 1


    hands_thresholds = {
        'left': ManhattanDistance(preds['left_hip'][:2], preds['left_wrist'][:2]) < ManhattanDistance(preds['right_hip'][:2], preds['left_wrist'][:2]),
        'right': ManhattanDistance(preds['right_hip'][:2], preds['right_wrist'][:2]) < ManhattanDistance(preds['left_hip'][:2], preds['right_wrist'][:2]),
    }

    closer = None
    min_distance = None

    current_hand = None
    threshold = None
    for i in reversed(range(len(seq_array_test))):

        if seq_array_test[i] in ['ear', 'shoulder']:
            threshold = ear_threshold
        else:
            threshold = hip_threshold

        ld = Distance(preds[f'right_{seq_array_test[i]}'][:2], preds['left_wrist'][:2])
        rd = Distance(preds[f'left_{seq_array_test[i]}'][:2], preds['right_wrist'][:2])

        if (min(ld, rd) < threshold) and not (hands_thresholds['left'] and hands_thresholds['right']):

            if counter is None:
                counter = create_closer(seq_array_test)

            empty_frames = 0

            if (min_distance is None) or (min_distance > min(ld, rd)):
                min_distance = min(ld, rd)
                if ld > rd:
                    closer = f'left_{seq_array_test[i]}'
                    current_hand = 'right'
                else:
                    closer = f'right_{seq_array_test[i]}'
                    current_hand = 'left'

    if closer is not None:
        counter[closer] += 1
        direc = closer.split('_')[0]
        if (hand is None) or ((current_hand is not None) and (hand != current_hand) and hands_thresholds[hand]): #(len(detection_directions) == 0) or (detection_directions[-1] != direc):
            detection_directions.append(direc)
            _det_time = frame_count/fps
            touche_time.append(_det_time)
            hand = current_hand

            ######## Score 2 ########
            _ind = np.argwhere((consider_time[:, 0] <= _det_time) & (_det_time <= consider_time[:, 1]))
            if (len(_ind) > 0) and (time_scores_counted[_ind[0, 0]] < 3):
                score_2 += 1
                time_scores_counted[_ind[0, 0]] += 1
                if args.log:
                    print('Score 2:', score_2, 'time:', round(_det_time, 2))

    if ((hands_thresholds['left'] and hands_thresholds['right']) or (frame_count == total_frames))\
            and (counter is not None):
        if (empty_frames < fps) and (frame_count != total_frames):
            empty_frames += 1
        else:

            counter_acc = dict(zip(seq_array_test, [0]*len(seq_array_test)))
            for key, value in counter.items():
                if key.split('_')[1] in seq_array_test:
                    counter_acc[key.split('_')[1]] += value

            touched = None
            if ((counter.get('left_ear', 0) > 0) and (counter.get('right_ear', 0) > 0)) \
                    and ((counter.get('left_shoulder', 0) == 0) or (counter.get('right_shoulder', 0) == 0)):
                touched = 'ear'
            elif ((counter.get('left_ear', 0) == 0) or (counter.get('right_ear', 0) == 0)) \
                    and ((counter.get('left_shoulder', 0) > 0) and (counter.get('right_shoulder', 0) > 0)):
                touched = 'shoulder'
            elif ((counter.get('left_hip', 0) == 0) or (counter.get('right_hip', 0) == 0)) \
                    and ((counter.get('left_knee', 0) > 0) and (counter.get('right_knee', 0) > 0)):
                touched = 'knee'
            elif counter_acc.get('ear', 0) >= 6:
                touched = 'ear'
            elif counter_acc.get('shoulder', 0) >= 6:
                touched = 'shoulder'
            elif counter_acc.get('knee', 0) >= 6:
                touched = 'knee'
            elif counter_acc.get(f'hip', 0) >= 6:
                touched = 'hip'
            else:
                touched = max(counter_acc, key=lambda x: counter_acc[x])

            if touched is None:
                continue

            ####### Score 1 ########
            correct_touch = False
            if len(score_1_seq_array) > 0:
                correct_touch = seq_array_test[current_ind] == touched
                if correct_touch:
                    score_1 += 1
                    if args.log:
                        print('Score 1:', score_1, 'body part:', touched)
                score_1_seq_array.pop(0)
                # score_1_seq_array.remove(seq_array_test[current_ind])

            ###### Score 3 #######
            if correct_touch:
                score_3_counts = 0
                for _i, _correct_touch in enumerate(find_crrect_body_parts(detection_directions)):
                    if _correct_touch:
                        score_3 += 1
                        score_3_counts += 1
                        if args.log:
                            print('Score 3:', score_3, 'body part:', detection_directions[_i], touched)
                    if score_3_counts == 3:
                        break

            counter = None
            hand = None
            detection_frame_counts = 0
            empty_frames = 0
            touche_time = list()
            detection_directions = list()
            current_ind += 1


cap.release()
if args.visualize:
    cv2.destroyAllWindows()

print(f'''
Final Calculated Scores are:
    SCORE 1: {score_1}
    SCORE 2: {score_2}
    SCORE 3: {score_3}
    Total:   {score_1 + score_2 + score_3}
''')

print("Total time taken:", round((time() - st_time)/60), 'mins')





