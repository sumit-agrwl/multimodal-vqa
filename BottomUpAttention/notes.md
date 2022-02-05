# extract features from predetermined bounding boxes (validation)
python3 extract_features.py --mode caffe --num-cpus 0 --extract-mode bbox_feats --config-file configs/bua-caffe/extract-bua-caffe-r152.yaml --image-dir /home/ubuntu/scene_graph_generation/subsetImages/validation --bbox-dir /home/ubuntu/scene_graph_generation/attribute_prediction/predetermined_bboxes/validation --out-dir /home/ubuntu/scene_graph_generation/attribute_prediction/validation  --resume

# extract features from predetermined bounding boxes (training)
python3 extract_features.py --mode caffe --num-cpus 0 --extract-mode bbox_feats --config-file configs/bua-caffe/extract-bua-caffe-r152.yaml --image-dir /home/ubuntu/scene_graph_generation/subsetImages/training --bbox-dir /home/ubuntu/scene_graph_generation/attribute_prediction/predetermined_bboxes/training --out-dir /home/ubuntu/scene_graph_generation/attribute_prediction/training  --resume

# extract features from predetermined bounding boxes (training_fullset)
python3 extract_features.py --mode caffe --num-cpus 0 --extract-mode bbox_feats --config-file configs/bua-caffe/extract-bua-caffe-r152.yaml --image-dir /home/ubuntu/scene_graph_generation/subsetImages/training --bbox-dir /home/ubuntu/scene_graph_generation/attribute_prediction/predetermined_bboxes/training_fullset --out-dir /home/ubuntu/scene_graph_generation/attribute_prediction/training_full  --resume

# extract features from predetermined bounding boxes (testdev)
python3 extract_features.py --mode caffe --num-cpus 0 --extract-mode bbox_feats --config-file configs/bua-caffe/extract-bua-caffe-r152.yaml --image-dir /home/ubuntu/scene_graph_generation/subsetImages/testdev --bbox-dir /home/ubuntu/scene_graph_generation/attribute_prediction/predetermined_bboxes/testdev --out-dir /home/ubuntu/scene_graph_generation/attribute_prediction/testdev  --resume

# extract features from images (validation)
python3 extract_features.py --mode caffe --num-cpus 0 --extract-mode roi_feats --config-file configs/bua-caffe/extract-bua-caffe-r152.yaml --image-dir /home/ubuntu/scene_graph_generation/subsetImages/validation --out-dir /home/ubuntu/scene_graph_generation/attribute_prediction/validation  --resume

# extract features from predetermined bounding boxes (all)
python3 extract_features.py --mode caffe --num-cpus 0 --extract-mode bbox_feats --config-file configs/bua-caffe/extract-bua-caffe-r152.yaml --image-dir /media/fxx/Large_Store/multimodal_ml/gqa_dataset/allImages/images --bbox-dir /media/fxx/Large_Store/multimodal_ml/gqa_dataset/sceneGraphs/all_sgg_generated_organized/bboxes --out-dir /media/fxx/Large_Store/multimodal_ml/gqa_dataset/sceneGraphs/all_sgg_generated_organized/attribute_generated_scene_graphs  --resume