import os
import json
from detectron2.structures import Boxes
import pickle
from PIL import Image
import numpy as np

meta_info_dir = "/media/fxx/Large_Store/multimodal_ml/code/attribute_prediction/bottom-up-attention.pytorch/meta_info"

# open attributes.json file
with open(os.path.join(meta_info_dir, "attributes.json"), 'r') as outfile:
    attributes_info = json.load(outfile)

# open predicates.json file
with open(os.path.join(meta_info_dir, "predicates.json"), 'r') as outfile:
    predicates_info = json.load(outfile)

# open attr_gqa.txt file
with open(os.path.join(meta_info_dir, "attr_gqa.txt"), 'r') as outfile:
    attr_gqa_data = outfile.read().splitlines()

# open attribute_hierarchy.json file
with open(os.path.join(meta_info_dir, "attribute_hierarchy.json"), 'r') as outfile:
    attribute_hierarchy_info = json.load(outfile)

# open obj2attribute.json file
with open(os.path.join(meta_info_dir, "obj2attribute.json"), 'r') as outfile:
    obj2attribute_info = json.load(outfile)
    
attribute_txt_info_dir = "/media/fxx/Large_Store/multimodal_ml/code/attribute_prediction/bottom-up-attention.pytorch/evaluation"
# open attribute_object vocab
with open(os.path.join(attribute_txt_info_dir, "objects_vocab.txt"), 'r') as outfile:
    attributes_txt_object_vocab_info = outfile.read().splitlines()

# open attribute_attributes vocab
with open(os.path.join(attribute_txt_info_dir, "attributes_vocab.txt"), 'r') as outfile:
    attributes_txt_attributes_vocab_info = outfile.read().splitlines()

# open base generated scene graphs json file
dataset_type = "fullset"
base_scenegraphs_dir = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/sceneGraphs/all_sgg_generated_organized/generated_"+dataset_type+"_sceneGraphs.json"
with open(os.path.join(base_scenegraphs_dir), 'r') as outfile:
    base_scenegraphs_data = json.load(outfile)

updated_scenegraphs_data = base_scenegraphs_data
# set folder where attributes are located .npz files
attributes_predicted_dir = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/sceneGraphs/all_sgg_generated_organized/attribute_generated_scene_graphs"

dataset_size = len(base_scenegraphs_data)

image_dir = "/media/fxx/Large_Store/multimodal_ml/gqa_dataset/allImages/images"
count = 0
for image_id, data in base_scenegraphs_data.items():
    print("On count: " + str(count))
    count += 1
    # open predicted attributes file
    # if count >= 39027 and image_id[0] == '_':
    #     updated_scenegraphs_data[image_id[15:]] = updated_scenegraphs_data[image_id]
    #     del updated_scenegraphs_data[image_id]
    #     image_id = image_id[15:]
    #     predicted_attributes_image = np.load(os.path.join(attributes_predicted_dir,str(image_id)+'.npz'), allow_pickle=True)
    # else:    
    predicted_attributes_image = np.load(os.path.join(attributes_predicted_dir,str(image_id)+'.npz'), allow_pickle=True)
    
    # extract information from .npz file
    image_height = predicted_attributes_image['info'].item()['image_h']
    image_width = predicted_attributes_image['info'].item()['image_w']
    image_num_bboxes = predicted_attributes_image['info'].item()['num_boxes']
    image_objects_id = predicted_attributes_image['info'].item()['objects_id']
    image_objects_conf = predicted_attributes_image['info'].item()['objects_conf']
    image_attrs_id = predicted_attributes_image['info'].item()['attrs_id']
    image_attrs_conf = predicted_attributes_image['info'].item()['attrs_conf']

    # update image height and width
    updated_scenegraphs_data[image_id]['height'] = image_height
    updated_scenegraphs_data[image_id]['width'] = image_width
    
    # FORNOW, check that the object detection of the attribute prediction is equal to the object detection of the 
    scene_graph_objects_list = list(updated_scenegraphs_data[image_id]['objects'].keys())
    for idx in range(len(image_objects_id)):
        predicted_attributes_object_name = attributes_txt_object_vocab_info[image_objects_id[idx]]
        scene_graph_object_name = updated_scenegraphs_data[image_id]['objects'][scene_graph_objects_list[idx]]['name']
        if predicted_attributes_object_name == scene_graph_object_name or predicted_attributes_object_name[:-1] == scene_graph_object_name :
            if attributes_txt_attributes_vocab_info[image_attrs_id[idx]] in attributes_info:
                updated_scenegraphs_data[image_id]['objects'][scene_graph_objects_list[idx]]['attributes'].append(attributes_txt_attributes_vocab_info[image_attrs_id[idx]])

# import pdb; pdb.set_trace()
print("Done.")