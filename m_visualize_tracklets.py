#!/usr/bin/env python

"""
DO NOT MODIFY THIS FILE! DO NOT MODIFY THIS FILE! DO NOT MODIFY THIS FILE!

For HW5 of CSE353 Machine Learning Fall 2020
This code will create an output video for visualizing vehicle tracklets

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 26-Oct-2020
Last modified: 26-Oct-2020
"""

import argparse
import json
import cv2
import random
from tqdm import tqdm
import os

def disp_vehicles(vehicle_file, input_video, output_video):
    with open(vehicle_file) as f:
        vehicle_data = json.load(f)

    disp_data = {}
    max_cluster_id = 0

    print("Processing tracklets")
    for v_id in tqdm(vehicle_data):
        v_tracklet = vehicle_data[v_id]
        v_class = v_tracklet["class"]
        if "direction_id" in v_tracklet:
            v_cluster_id = v_tracklet["direction_id"]
            assert v_cluster_id >=0, "direction_id cannot be negative"
            max_cluster_id = max(max_cluster_id, v_cluster_id)
        else:
            v_cluster_id = 0

        for v_det in v_tracklet["tracks"]:
            frm_id = v_det[0]
            bbox = v_det[1:]

            disp_info = [v_id, v_class, v_cluster_id, bbox]

            if frm_id in disp_data:
                disp_data[frm_id].append(disp_info)
            else:
                disp_data[frm_id] = [disp_info]

    cam = cv2.VideoCapture(input_video)
    fps = cam.get(cv2.CAP_PROP_FPS)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video, fourcc, int(fps), (int(width), int(height)))
    num_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # create colormap for clusters
    num_cluster = max_cluster_id + 1
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)]
    if num_cluster > 7: # add random colors
        for i in range(7, num_cluster):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            colors.append(color)

    print("Rendering video frames")
    frm_id = 0
    for _ in tqdm(range(num_frames)):
        _, im = cam.read()
        if im is None:
            continue
        frm_id += 1

        if frm_id in disp_data:
            vehicles2disp = disp_data[frm_id]

            for vehicle in vehicles2disp:
                bbox = vehicle[3]
                bbox2 = [int(x) for x in bbox]
                cluster_id = vehicle[2]
                color = colors[cluster_id]
                start_point = (bbox2[0], bbox2[1])
                end_point = (bbox2[2], bbox2[3])
                thickness = 2
                im = cv2.rectangle(im, start_point, end_point, color, thickness)
                if cluster_id == 0:
                    im = cv2.putText(im, "Outlier: {} {}".format(vehicle[1], vehicle[0]), (bbox2[0], bbox2[1]),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    im = cv2.putText(im, "{} {}".format(vehicle[1], vehicle[0]), (bbox2[0], bbox2[1]),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out_video.write(im)

    out_video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate output video file to visualize tracklets")
    parser.add_argument("-iv", "--input_video", type=str, required=True, help="/Path/to/input/video/file/")
    parser.add_argument("-ov", "--output_video", type=str, help="/path/to/output/video/file")
    parser.add_argument("-t",  "--tracklet_file", type=str, required=True, help="/Path/to/input/json/file/of/tracking/result")
    args = parser.parse_args()

    if args.output_video is None:
        dir_path = os.path.dirname(args.input_video)
        output_video = "{}/out_{}".format(dir_path, os.path.basename(args.input_video))
    else:
        output_video = args.output_video

    disp_vehicles(args.tracklet_file, args.input_video, output_video)
