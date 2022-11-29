#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import imgviz
import numpy as np

import labelme
import time
import cv2
import pose_config

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def images_init () :
    '''
    一张图像的初始化信息
    '''
    a = {'id': -1, 'width': -1, 'height': -1, 'file_name': '', 'license': -1, 'flickr_url': '', 'coco_url': '', 'date_captured': ''}
    return a

def perpare_images_info(cur_img):
    img_data = cv2.imread(cur_img, cv2.IMREAD_COLOR)

    img_info = images_init()
    img_info['id'] = int(os.path.basename(cur_img).split(".")[0])
    img_info['width'] = img_data.shape[1]
    img_info['height'] = img_data.shape[0]
    img_info['file_name'] = os.path.basename(cur_img)
    return img_info


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"))
    print("Creating dataset:", args.output_dir)


    out_ann_file = osp.join(args.output_dir, "person_keypoints_train2017.json")
    os.popen('cp {} {}'.format("template.json", out_ann_file))
    time.sleep(2)
    print("finish copy template json file")

    with open(out_ann_file, 'r') as load_f:
        dataset = json.load(load_f)
    print('-- images: %6d    anno: %6d | ' % (len(dataset['images']), len(dataset['annotations'])))


    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    num_img = 0
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)
        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        in_img_file = osp.join(args.input_dir, base + ".jpg")

        if not (os.path.exists(in_img_file)):
            print("!!Error file not exist", in_img_file)
            continue

        os.popen('cp {} {}'.format(in_img_file, out_img_file))

        print("to load file ", filename)
        label_file = labelme.LabelFile(filename=filename)

        img_info = perpare_images_info(in_img_file)
        dataset['images'].append(img_info)
        max_iscrowd_id = 10000

        anno = {}
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type")

            if group_id is None and shape_type != "polygon":
                print("**************************************** ignore ungroup shape", shape)
                continue


            if group_id not in anno.keys() and group_id is not None:
                anno_id = int(base) * 100000 + group_id + 1
                anno[group_id] = {"num_keypoints": 0, "iscrowd": 0,
                     "keypoints": [ 0 for i in range(17*3)], "image_id": int(base) ,
                     "bbox": [], "category_id": 1, "id": anno_id, "area": -1,
                     "segmentation": []}


            if shape_type == "pose":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([int(x1), int(x2)])
                y1, y2 = sorted([int(y1), int(y2)])

                anno[group_id]["bbox"] = [x1,y1,x2-x1,y2-y1]
                anno[group_id]["area"] = (x2-x1) *(y2-y1)

            elif shape_type == "polygon" and label =="person":
                if group_id is not None:
                    points = np.asarray(points).flatten().tolist()
                    points = [int(x) for x in points]
                    anno[group_id]["segmentation"].append(points)
                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!ignore not support segmentation", shape)

            elif shape_type == "point":
                keypoints = pose_config.pose_define["keypoints"]
                if keypoints.count(label) > 0 :
                    index = keypoints.index(label)
                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!find point label not in list", shape)
                    continue
                (x, y) = points[0]
                if index >=0:
                    anno[group_id]["keypoints"][index*3:(index+1)*3] = [int(x),int(y),2]
                    anno[group_id]["num_keypoints"] = anno[group_id]["num_keypoints"] + 1


            elif shape_type == "polygon" and label == "iscrowd":
                anno_id = int(base) * 100000 + max_iscrowd_id + 1
                bbox = cv2.boundingRect(np.array(points).astype(int))
                anno[max_iscrowd_id] = {
                    "num_keypoints": 0, "iscrowd": 1,
                    "keypoints": [0 for i in range(17 * 3)], "image_id": int(base),
                    "bbox": bbox, "category_id": 1, "id": anno_id, "area": bbox[2]*bbox[3],
                    "segmentation": [[int(x) for x in np.asarray(points).flatten().tolist()]]
                }
                max_iscrowd_id += 1

            else:
                print("ignore unsupport shape", shape)

        num_img += 1
        print('-- images: %6d    anno: %6d     %5d    |    %s' % (
            len(dataset['images']), len(dataset['annotations']), num_img, filename))

        for key, info in anno.items():
            dataset['annotations'].append(info)










    with open(out_ann_file, "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    main()
