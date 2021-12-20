# -*- coding: utf-8 -*-

import cv2
import argparse
import numpy as np
import os,sys
from pathlib import Path
from piplines.detectors import ChristmasDetection
from motpy import Detection, MultiObjectTracker
from motpy.testing_viz import draw_detection, draw_track
import collections

def get_script_path():
    """
        Get the path to the current script
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def run_camera(source=0, save=False, show_tracking = True, show_objects = False):
    """
        Starting the camera with detection
    """

    #============================== Main settings ================================
    
    # DEBUG
    # show_tracking=True 
    # show_objects=False
    # source = str(Path(get_script_path(), 'data', 'sample.mp4'))

    # classes and models
    obj_classes = {
        40: ['wine glass', [213,134,24],  'Wine Glass'],
        46: ['banana', [101,212,243],  'Banana'],
        49: ['orange', [66,160,239],  'Tangerine'],
        39: ['bottle', [101,194,116],  "Let's open! ;)"],
    }
    obj_wights = str(Path(get_script_path(), 'weights', 'yolov5m6.pt'))
    
    # detector

    detector_conf_thres  = 0.2  # minimum prediction coeff 
    detector_iou_thres   = 0.7  # confidence in the object
    detector_input_width = 1280  # input_width - the size of the image. affects quality and stability, but reduces performance when larges

    #tracker

    tracker_spec = {
        'order_pos': 1, 
          'dim_pos': 2,         # position is a center in 2D space; under constant velocity model
       'order_size': 0, 
         'dim_size': 2,         # bounding box is 2 dimensional; under constant velocity model
        'q_var_pos': 1000,      # process noise
        'r_var_pos': 0.1        # measurement noise
    }
    tracker_kwargs = {
        'max_staleness': 5
    }
    tracker_min_steps_alive = 30
     
     #============================== /Main settings ===============================

    # detection
    road_users_detector = ChristmasDetection(obj_wights, obj_classes, conf_thres=detector_conf_thres, 
        iou_thres=detector_iou_thres, input_width = detector_input_width)

    # plotting
    cap = cv2.VideoCapture(source)

    # recorder
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920,1080))
    
    fps = float(cap.get(cv2.CAP_PROP_FPS)) 

    # tracker
    tracker = MultiObjectTracker(dt=0.1 ,# 1/video_fps, 
        tracker_kwargs=tracker_kwargs, model_spec=tracker_spec)

    counter = []

    if cap.isOpened():
        window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
        
        # Window
        while cv2.getWindowProperty("Camera", 0) >= 0:
            ret, frame = cap.read()

            if ret:

                # detection process
                objs = road_users_detector.detect(frame)

                # plotting
                detections = []

                if show_objects:
                    for obj in objs:
                        title = obj['title']
                        score = obj['score']
                        color = obj['color']
                        [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
                        cv2.putText(frame, f'{title} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)
                
                # show tracker

                if show_tracking:
                    for obj in objs:
                        [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                        detections.append(Detection(box=[xmin,ymin,xmax,ymax], score=obj['score'], payload=obj))

                    _ = tracker.step(detections=detections)
                    tracks = tracker.active_tracks(min_steps_alive=tracker_min_steps_alive)

                    
                    orange_counter = 0

                    # Tangerines counter

                    for track in tracks:
                        cv2.rectangle(frame, (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])), track.payload['color'], 2)
                        cv2.putText(frame,  str(track.payload['title']) + f' {track.id[:5]}', (int(track.box[0]), int(track.box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, track.payload['color'],2, cv2.LINE_AA)
                        
                        if track.payload['label'] == 'orange':
                            orange_counter = orange_counter + 1

                    cv2.putText(frame, 'Tangerines: ' + str(orange_counter), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                       

            out.write(frame)
            cv2.imshow("Camera", frame)

            keyCode = cv2.waitKey(1)
            if keyCode == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


def parse_params():
    """
        Parsing the parameters of the request to the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type = int, default = 0 , help = 'source (camera number, video file, etc.)')
    args = parser.parse_args()
    return args


def main(opt):
    """
        Main fuction
    """
    run_camera(**vars(opt))


if __name__ == "__main__":
    """
        Constructor
    """
    opt = parse_params()
    main(opt)
