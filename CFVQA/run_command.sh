CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$PYTHONPATH:src python src/gqa/gqa_run.py \
--train train --valid val --loadLXMERTQA snap/pretrained/model \
--loadGraphVQA snap/pretrained/pretrained_10_epochs.pth --batchSize 256 \
--epochs 15 --logFile output_train.log --tqdm --multiGPU