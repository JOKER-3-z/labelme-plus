# coding=utf-8
import os
import sys
import numpy as np
import json
import copy
import cv2
import random
import shutil

here = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(here, 'output/JPEGImages/')
noLR = False

if len(sys.argv) > 1:
    json_file = sys.argv[1]
else:
    json_file = os.path.join(here, "output/person_keypoints_train2017.json")

with open(json_file, 'r') as load_f:
    annoRes = json.load(load_f)

dataset = {'images': [],
           'annotations': [],
           'categories': []
           }
dataset['categories'] = [{"supercategory": "person",
                          "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                                       [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6],
                                       [5, 7]], "id": 1,
                          "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                                        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                                        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
                                        "right_ankle"], "name": "person"}]

try:
    dataset['annotations'] = annoRes['annotations']
    dataset['images'] = list(set([ann['image_id'] for ann in annoRes['annotations']]))
except:
    dataset['annotations'] = annoRes
    dataset['images'] = list(set([ann['image_id'] for ann in annoRes]))

num_img = len(dataset['images'])
num_ann = len(dataset['annotations'])
print('-- images: %6d    anno: %6d | ' % (num_img, num_ann))
# cv2.namedWindow('win', cv2.WINDOW_NORMAL)


# cv2.setWindowProperty ('win', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def run(keywords=''):
    i = 0
    print(dataset['images'])
    while (1):
        image_info = dataset['images'][i]
        print(image_info)
        path = os.path.join(image_path, "" + '%08d'%(image_info)+ ".jpg")
        if not os.path.exists(path):
            raise ValueError('[ERROR] file not exists. %s' % path)
        image_data = cv2.imread(path, cv2.IMREAD_COLOR)
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        # d = np.random.normal(5, 8, image_data.shape)
        # image_data = image_data + d
        # image_data = np.clip(image_data,0,255)
        # image_data = image_data.astype(np.uint8)

        annIds = []
        for j in range(len(dataset['annotations'])):
            if image_info == dataset['annotations'][j]['image_id']:
                annIds.append(j)

        more_str = '%d/%d %d' % (i, num_img, image_info)
        # print annIds
        show(image_data, annIds, more_str)
        # c = cv2.waitKey(0)
        # print("get char ", c)
        # if c == 27:
        #     break
        # elif c == ord('a') or c == 1048673:
        #     i -= 1
        #     if i < 0:
        #         i = num_img - 1
        # elif c == ord('d') or c == 1048676:
        i += 1
        print("i is ", i)
        if i >= num_img:
            break
        # show(image_data, annIds, more_str)
        # print image_info
        # for i in range(len(annIds)) :
        #    print dataset['annotations'][annIds[i]]


def check_one():
    for i in range(len(dataset['images'])):
        image_info = dataset['images'][i]
        fg0 = image_info['file_name'].rfind('458')
        if fg0 >= 0:
            path = os.path.join(image_path, image_info['file_name'])
            if not os.path.exists(path):
                raise ValueError('[ERROR] file not exists. %s' % path)
            image_data = cv2.imread(path, cv2.IMREAD_COLOR)
            annIds = []
            for j in range(len(dataset['annotations'])):
                if image_info['id'] == dataset['annotations'][j]['image_id']:
                    annIds.append(j)

            more_str = '%d/%d %d' % (i, num_img, image_info['id'])
            # print annIds
            show(image_data, annIds, more_str)
            c = cv2.waitKey(0)


def show(showingImg, annIds, more_str):
    for i in range(len(annIds)):
        ann = dataset['annotations'][annIds[i]]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # 画骨骼点的圈圈
        key_pts = ann['keypoints']
        for k in range(0, len(key_pts), 3):
            # 鼻子部分左右，用绿色表示
            if k == 0:
                clr = (0, 255, 0)
                sss = ''
            # 奇数为左（蓝色表示），偶数为右(红色表示)
            elif k % 2 == 0:
                clr = (0, 0, 255)  # right, red
                sss = 'R'
            else:
                clr = (255, 0, 0)  # left, blue
                sss = 'L'
            if noLR:
                clr = color
                sss = ''
            r = 6
            if key_pts[k + 2] == 2:
                cv2.circle(showingImg, (int(key_pts[k]), int(key_pts[k + 1])), r, clr, -1)
            elif key_pts[k + 2] == 1:
                cv2.circle(showingImg, (int(key_pts[k]), int(key_pts[k + 1])), r, clr, 2)
            cv2.putText(showingImg, sss, (int(key_pts[k]) - r - 1, int(key_pts[k + 1]) - r - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
        # 画骨骼点之间的连线
        ske = dataset['categories'][0]

        for s in ske['skeleton']:
            idx0 = s[0] - 1
            idx1 = s[1] - 1
            if (ann['keypoints'][idx0 * 3 + 0] > 0 and \
                    ann['keypoints'][idx0 * 3 + 1] > 0 and \
                    ann['keypoints'][idx0 * 3 + 2] > 0 and \
                    ann['keypoints'][idx1 * 3 + 0] > 0 and \
                    ann['keypoints'][idx1 * 3 + 1] > 0 and \
                    ann['keypoints'][idx1 * 3 + 2] > 0):
                cv2.line(showingImg, (int(ann['keypoints'][idx0 * 3 + 0]), int(ann['keypoints'][idx0 * 3 + 1])),
                         (int(ann['keypoints'][idx1 * 3 + 0]), int(ann['keypoints'][idx1 * 3 + 1])), color, 2)
        # 显示轮廓
        try:
            for cur_poly in ann['segmentation']:
                l = len(cur_poly)
                pt0 = (int(cur_poly[l - 2]), int(cur_poly[l - 1]))
                for j in range(0, l, 2):
                    pt1 = (int(cur_poly[j]), int(cur_poly[j + 1]))
                    cv2.line(showingImg, pt0, pt1, (255, 0, 255), 1)
                    pt0 = pt1
        except:
            print("no segmentation")

        # 显示外接矩形
        try:
            color = (0, 255, 255)
            pt0 = (int(ann['bbox'][0]), int(ann['bbox'][1]))
            pt1 = (int(ann['bbox'][0]) + int(ann['bbox'][2]), int(ann['bbox'][1]) + int(ann['bbox'][3]))
            cv2.rectangle(showingImg, pt0, pt1, color, 1)
            print(ann['id'],ann['num_keypoints'])
            cv2.putText(showingImg, '%012d-%d' % (ann['id'], ann['num_keypoints']), (pt0[0], pt0[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except:
            print("no bbox")
    cv2.putText(showingImg, more_str, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imwrite("{}_show.jpg".format(annIds), showingImg)


if __name__ == '__main__':
    # check_one ()
    run()