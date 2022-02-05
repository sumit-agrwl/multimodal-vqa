import os
import json
import pickle

import numpy as np
from numpy.lib.type_check import imag

if __name__ == "__main__":
    # /media/fengxiang/linux_storage_ba/multimodal_ml/datasets/upload_causal_motif_sgdet/gqa_subset_testdev_detected
    sgg_dir  = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/sceneGraphs/all_sgg_generated_unorganized"
    other_info_dir = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/sceneGraphs"

    # load info files about predictions
    print("Loading custom info about generated scene graphs")
    with open(os.path.join(sgg_dir, "custom_data_info.json")) as file:
        subset_info = json.load(file)

    # load validation scene graphs file
    print("Opening ground truth training scene graphs")
    subset_scenegraphs_dir = os.path.join(other_info_dir,"train_sceneGraphs.json") 
    with open(subset_scenegraphs_dir) as file:
        val_subset_sg = json.load(file)

    # load directory of classifications for ground truth scene graphs
    gqa_label2ans_info_dir = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/meta_info/trainval_label2ans.json"
    print("Opening GQA ground truth information")
    with open(gqa_label2ans_info_dir) as file:
        gqa_label2ans_info = json.load(file)

    gqa_ans2label_info_dir = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/meta_info/trainval_ans2label.json"
    print("Opening GQA ground truth information")
    with open(gqa_ans2label_info_dir) as file:
        gqa_ans2label_info = json.load(file)

    relations_info_dir = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/meta_info/rel_gqa.txt"
    print("Opening relations text information")
    with open(relations_info_dir) as f:
        gqa_rel_info_numbered = f.read().splitlines()

    gqa_rel_info = []
    for item in gqa_rel_info_numbered[1:]:
        gqa_rel_info.append(item)

    # shape predictions files to be the same format
    '''
    format:
    val_sg.json = {
        "image_id": {
            'width': int <image>
            'objects': {
                'unique_object_id': {
                    'name': str
                    'h': int <bounding box>
                    'relations': [ <array>
                        {
                            'object': 'unique_object_id'
                            'name': 'relation'
                        }, ...
                    ]
                    'w': int <bounding box>
                    'y': int <bounding box center>
                    'x': int <bounding box center>
                } 
            }
            'location': str
            'height: int <image>
        }
    }
    '''
    # define hyperparameters
    object_num_max_thresh = 15 # if no object scores meet the minimum threshold, then get the top N objects
    object_num_min = 4
    object_num_min_score_thresh = 0.3 # minimum score to consider objects
    # stop considering objects if minimum score or max number of objects is reached, whichever is first

    object_rel_thresh = 0.3 # minimum relation threshold to consider

    generated_sgg_filename = "train_part1_sgg_json.json"

    generated_sgg_dict = {}


    nameNotFound_dict = {}
    # create dictionary of names that are not compatible in the dictionary
    for idx in range(len(subset_info['ind_to_classes'])):
        if subset_info['ind_to_classes'][idx] not in gqa_ans2label_info.keys():
            if subset_info['ind_to_classes'][idx] not in nameNotFound_dict.keys():
                nameNotFound_dict[idx] = subset_info['ind_to_classes'][idx]


    relationNotFound_dict = {}
    # create dictionary of relations that are not compatible in the dictionary
    for idx in range(len(subset_info['ind_to_predicates'])):
        if subset_info['ind_to_predicates'][idx] not in gqa_rel_info:
            if subset_info['ind_to_predicates'][idx] not in relationNotFound_dict.keys():
                relationNotFound_dict[idx] = subset_info['ind_to_predicates'][idx]

    dataset_image_size = 148854
    nameNotFound_count = 0
    noObject_count = 0
    for count in range(dataset_image_size):
        # load predictions file
        print("Opening testdev scene graph predictions | " + str(count))
        with open(os.path.join(sgg_dir, "custom_prediction_"+str(count)+".json")) as file:
            validation_subset_predictions = json.load(file)
    
    # for idx, dictInfo in validation_subset_predictions.items():
        # find image_id from index
        key_idx = list(validation_subset_predictions.keys())[0]
        dictInfo = list(validation_subset_predictions.values())[0]
        image_id = subset_info['idx_to_files'][count][66:-4]
        generated_sgg_dict[image_id] = {}

        # create unique labelling objects to unique_object_id's (already done in )

        # acquire list of objects
        # (xmin, ymin, xmax, ymax)
        object_bboxes = np.array(dictInfo['bbox'])
        object_labels = np.array(dictInfo['bbox_labels'])
        object_scores = np.array(dictInfo['bbox_scores'])
        object_index = np.arange(len(object_scores))

        # get num objects larger than score
        object_scores_thresh = object_scores > object_num_min_score_thresh

        # set object labels that are not found to False
        for idx in object_index[object_scores_thresh]:
            if object_labels[idx] in nameNotFound_dict.keys():
                object_scores_thresh[idx] = False

        object_scores_thresh_sum = np.sum(object_scores_thresh)

        if object_num_max_thresh < object_scores_thresh_sum:
            object_topN = object_scores > object_scores[object_num_max_thresh-1]
        else:
            object_topN = object_scores > 0.0

        for idx in object_index[object_topN]:
            if object_labels[idx] in nameNotFound_dict.keys():
                object_topN[idx] = False

        if np.sum(object_topN) > object_num_max_thresh:
            topNCount = 0
            for topNIdx in range(len(object_topN)):
                if object_topN[topNIdx] is True:
                    topNCount += 1
                if topNCount > object_num_min:
                    object_topN[topNIdx] = False

        # # get top N number of objects
        # object_scores_topN = object_scores[:object_num_max_thresh]

        # if there is a smaller number of score threshold objects, then make those the main, else just consider the top N objects
        if np.sum(object_scores_thresh) < np.sum(object_topN) and np.sum(object_scores_thresh) != 0:
            object_bboxes_thresholded = object_bboxes[object_scores_thresh]
            object_labels_thresholded = object_labels[object_scores_thresh]
            object_scores_thresholded = object_scores[object_scores_thresh]
            object_index_thresholded = object_index[object_scores_thresh]
        else:
            object_bboxes_thresholded = object_bboxes[object_topN]
            object_labels_thresholded = object_labels[object_topN]
            object_scores_thresholded = object_scores[object_topN]
            object_index_thresholded = object_index[object_topN]
        
        if len(object_index_thresholded) == 0:
            import pdb; pdb.set_trace()

        # map sgg labels to scene graph labels
        object_label_names_thresholded = np.array([])
        for idx in range(len(object_labels_thresholded)):
            # convert sgg label to name
            label_name_ = subset_info['ind_to_classes'][object_labels_thresholded[idx]]
            if label_name_ not in gqa_ans2label_info.keys():
                # import pdb; pdb.set_trace()
                nameNotFound_count += 1
                print(nameNotFound_count)
                object_label_names_thresholded = np.append(object_label_names_thresholded, 'NOTFOUND')
                # if label_name_ not in nameNotFound_dict.keys():
                #     nameNotFound_dict[label_name_] = object_labels_thresholded[idx]
                # else:
                #     nameNotFound_dict[label_name_] += 1
            else:
                object_label_names_thresholded = np.append(object_label_names_thresholded, label_name_)

        # TODO input image width value in later
        generated_sgg_dict[image_id]['width'] = None
        # TODO input image height value in later
        generated_sgg_dict[image_id]['height'] = None
        # TODO input image location in later
        generated_sgg_dict[image_id]['location'] = ""

        

        generated_sgg_dict[image_id]['objects'] = {}

        for idx in range(len(object_index_thresholded)):
            generated_sgg_dict[image_id]['objects'][str(object_index_thresholded[idx])] = {}
            generated_sgg_dict[image_id]['objects'][str(object_index_thresholded[idx])]['name'] = object_label_names_thresholded[idx]
            generated_sgg_dict[image_id]['objects'][str(object_index_thresholded[idx])]['h'] = int(object_bboxes_thresholded[idx][3] - object_bboxes_thresholded[idx][1])
            generated_sgg_dict[image_id]['objects'][str(object_index_thresholded[idx])]['w']  = int(object_bboxes_thresholded[idx][2] - object_bboxes_thresholded[idx][0])
            generated_sgg_dict[image_id]['objects'][str(object_index_thresholded[idx])]['y'] = int((object_bboxes_thresholded[idx][3] + object_bboxes_thresholded[idx][1]) / 2)
            generated_sgg_dict[image_id]['objects'][str(object_index_thresholded[idx])]['x'] = int((object_bboxes_thresholded[idx][2] + object_bboxes_thresholded[idx][0]) / 2)
            # TODO add object attributes later
            generated_sgg_dict[image_id]['objects'][str(object_index_thresholded[idx])]['attributes'] = []
            generated_sgg_dict[image_id]['objects'][str(object_index_thresholded[idx])]['relations'] = []
            for sub_idx in range(len(validation_subset_predictions[str(key_idx)]['rel_pairs'])):
                if validation_subset_predictions[str(key_idx)]['rel_pairs'][sub_idx][0] != object_index_thresholded[idx]:
                    continue
                if validation_subset_predictions[str(key_idx)]['rel_pairs'][sub_idx][1] not in object_index_thresholded:
                    continue
                if validation_subset_predictions[str(key_idx)]['rel_labels'][sub_idx] in relationNotFound_dict.keys():
                    continue
                if validation_subset_predictions[str(key_idx)]['rel_pairs'][sub_idx][0] == validation_subset_predictions[str(key_idx)]['rel_pairs'][sub_idx][1]:
                    continue
                if validation_subset_predictions[str(key_idx)]['rel_scores'][sub_idx] < object_rel_thresh:
                    break
                tempDict = {}
                tempDict['object'] = str(validation_subset_predictions[str(key_idx)]['rel_pairs'][sub_idx][1])
                tempRelName = subset_info['ind_to_predicates'][validation_subset_predictions[str(key_idx)]['rel_labels'][sub_idx]]
                tempDict['name'] = tempRelName
                generated_sgg_dict[image_id]['objects'][str(object_index_thresholded[idx])]['relations'].append(tempDict)



        # map classifications from predictions to the gqa classifications and input as name
        # pass bounding box prediction information into the dictionary
        # pass image information into the dictionary

    # output attribute detection for each object within the same model or find pretrained modesl that can output attributes for object bounding boxes
    # compute predictions for training and validation and compute metrics
    # train model on gqa subsets / datasets
    # due by the end of the week

    # save generated_sgg_dict as a .pkl file

    
    # with open(generated_sgg_filename, 'wb') as f:
    #     pickle.dump(generated_sgg_dict, f, pickle.HIGHEST_PROTOCOL)
    # with open(generated_sgg_filename, 'w') as outfile:
    #     json.dump(generated_sgg_dict, outfile)

    # import pdb; pdb.set_trace()
    print("Done.")

