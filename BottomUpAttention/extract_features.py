# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import os
import sys
import torch
# import tqdm
import cv2
import numpy as np
sys.path.append('detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances

from utils.utils import mkdir, save_features
from utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_bbox, save_roi_features
from utils.progress_bar import ProgressBar
from models import add_config
from models.bua.box_regression import BUABoxes

import ray
from ray.actor import ActorHandle

import json
from detectron2.structures import Boxes
import pickle
from PIL import Image


def switch_extract_mode(mode):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd

def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes, 
            'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def generate_npz(extract_mode, *args):
    if extract_mode == 1:
        save_roi_features(*args)
    elif extract_mode == 2:
        save_bbox(*args)
    elif extract_mode == 3:
        save_roi_features_by_bbox(*args)
    else:
        print('Invalid Extract Mode! ')

@ray.remote(num_gpus=1)
def extract_feat(split_idx, img_list, cfg, args, actor: ActorHandle):
    num_images = len(img_list)
    print('Number of images on split{}: {}.'.format(split_idx, num_images))

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()
    count = 0
    for im_file in (img_list):
        if os.path.exists(os.path.join(args.output_dir, im_file.split('.')[0]+'.npz')):
            actor.update.remote(1)
            continue
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        if im is None:
            print(os.path.join(args.image_dir, im_file), "is illegal!")
            actor.update.remote(1)
            continue
        dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
        # extract roi features
        if cfg.MODEL.BUA.EXTRACTOR.MODE == 1:
            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if not attr_scores is None:
                attr_scores = [attr_score.cpu() for attr_score in attr_scores]
            generate_npz(1, 
                args, cfg, im_file, im, dataset_dict, 
                boxes, scores, features_pooled, attr_scores)
        # extract bbox only
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
            with torch.set_grad_enabled(False):
                boxes, scores = model([dataset_dict])
            boxes = [box.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            generate_npz(2,
                args, cfg, im_file, im, dataset_dict, 
                boxes, scores)
        # extract roi features by bbox
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3:
            if not os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz')):
                actor.update.remote(1)
                continue
            count += 1
            # import pdb; pdb.set_trace()
            bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz'))['bbox']) # * dataset_dict['im_scale']
            if bbox.dim() == 1:
                bbox = bbox.unsqueeze(dim=0)
            proposals = Instances(dataset_dict['image'].shape[-2:])
            proposals.proposal_boxes = BUABoxes(bbox)
            dataset_dict['proposals'] = proposals

            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if not attr_scores is None:
                attr_scores = [attr_score.data.cpu() for attr_score in attr_scores]
            generate_npz(3, 
                args, cfg, im_file, im, dataset_dict, 
                boxes, scores, features_pooled, attr_scores)
            

        actor.update.remote(1)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/bua-caffe/extract-bua-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--num-cpus', default=1, type=int, 
                        help='number of cpus to use for ray, 0 means no limit')

    parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)

    parser.add_argument("--mode", default="caffe", type=str, help="bua_caffe, ...")

    parser.add_argument('--extract-mode', default='roi_feats', type=str,
                        help="'roi_feats', 'bboxes' and 'bbox_feats' indicates \
                        'extract roi features directly', 'extract bboxes only' and \
                        'extract roi features with pre-computed bboxes' respectively")

    parser.add_argument('--min-max-boxes', default='min_max_default', type=str, 
                        help='the number of min-max boxes of extractor')

    parser.add_argument('--out-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--image-dir', dest='image_dir',
                        help='directory with images',
                        default="image")
    parser.add_argument('--bbox-dir', dest='bbox_dir',
                        help='directory with bbox',
                        default="bbox")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = setup(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    num_gpus = len(args.gpu_id.split(','))

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    # Extract features.
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))

    if args.num_cpus != 0:
        ray.init(num_cpus=args.num_cpus)
    else:
        ray.init()
    img_lists = [imglist[i::num_gpus] for i in range(num_gpus)]

    pb = ProgressBar(len(imglist))
    actor = pb.actor

    print('Number of GPUs: {}.'.format(num_gpus))

    # convert bboxes into list of Box class boxes and save as .npz file
    json_base_dir = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/sceneGraphs/all_sgg_generated_organized"
    json_dataset_name = "fullset"
    json_file_name = "generated_"+json_dataset_name+"_noattributes_sceneGraphs.json"
    with open(os.path.join(json_base_dir, json_file_name), 'r') as outfile:
        json_file_data = json.load(outfile)
    # json_file_data = pickle.load(open(os.path.join(json_base_dir, json_file_name),'rb'))

    box_save_dir = os.path.join(json_base_dir, "bboxes")
    image_dir = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/allImages/images"

    count = 0
    noObject_count = []

    image_size_offset = 10

    for image_id, image_data in json_file_data.items():
        box_file_name = image_id+".npz"
        # TODO save a numpy array
        # box_numpy = torch.tensor([0,0,0,0]).unsqueeze(dim=0)
        box_numpy = np.array([0,0,0,0])[np.newaxis,:]
        if len(image_data['objects']) == 0:
            noObject_count.append(image_id)

        # create bounding box for image size
        # open image
        im = Image.open(os.path.join(image_dir, str(image_id)+".jpg"))
        # determine width and height of image
        im_width, im_height = im.size
        # add bounding box to image
        tempBox_y1 = image_size_offset
        tempBox_y2 = im_height-image_size_offset
        tempBox_x1 = image_size_offset
        tempBox_x2 = im_width - image_size_offset
        temp_box_numpy = np.array([tempBox_x1, tempBox_y1, tempBox_x2, tempBox_y2])[np.newaxis,:]
        box_numpy = np.append(box_numpy, temp_box_numpy, axis=0)

        for object_idx, object_data in image_data['objects'].items():
            tempBox_height = object_data['h']
            tempBox_width = object_data['w']
            tempBox_xCenter = object_data['x']
            tempBox_yCenter = object_data['y']
            
            tempBox_y1 = float(tempBox_yCenter - tempBox_height/2)
            tempBox_y2 = float(tempBox_yCenter + tempBox_height/2)
            tempBox_x1 = float(tempBox_xCenter - tempBox_width/2)
            tempBox_x2 = float(tempBox_xCenter + tempBox_width/2)

            # temp_box_numpy = torch.tensor([tempBox_x1, tempBox_y1, tempBox_x2, tempBox_y2]).unsqueeze(dim=0)
            temp_box_numpy = np.array([tempBox_x1, tempBox_y1, tempBox_x2, tempBox_y2])[np.newaxis,:]
            # box_numpy = torch.cat((box_numpy,temp_box_numpy), dim=0)
            box_numpy = np.append(box_numpy, temp_box_numpy, axis=0)

        box_numpy = box_numpy[1:]
        # if box_numpy.shape[0] == 0:
        #     const_offset = 10.0
        #     im = Image.open(os.path.join(image_dir, str(image_id)+".jpg"))
        #     im_width, im_height = im.size
        #     box_numpy = np.array([0.0+const_offset, 0.0+const_offset, im_width-const_offset, im_height-const_offset])
        # if box_numpy.shape[0] == 1:
        #     # box_numpy = torch.cat((box_numpy, box_numpy), dim=0)
        #     box_numpy = np.append(box_numpy,box_numpy, axis=0)
        save_box_object = box_numpy
        count += 1
        print(count)
        np.savez(os.path.join(box_save_dir,box_file_name), bbox=save_box_object)

    extract_feat_list = []
    for i in range(num_gpus):
        extract_feat_list.append(extract_feat.remote(i, img_lists[i], cfg, args, actor))
    
    # import pdb; pdb.set_trace()
    pb.print_until_done()
    ray.get(extract_feat_list)
    ray.get(actor.get_counter.remote())

if __name__ == "__main__":
    main()
