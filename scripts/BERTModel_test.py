import sys
import os
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



from transformers import pipeline
from transformers import AutoTokenizer, BertModel

import json
import csv
import torch



# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# feature_extractor = pipeline('feature-extraction', model=model_name, tokenizer = tokenizer)
# # run autotokenizer tokenize the question string
# input_batch_bert = tokenizer(questArr, padding=True, truncation=True, max_length=256)

# features_batch_bert = feature_extractor(questArr)


class modelBERT:

    def __init__(self,model_name, device_name):
        # device_name is either cpu or cuda:0
        self.model_name = model_name
        self.device_name = device_name
        self._padding = True
        self._truncation = True
        self._max_length = 256
        self.initializePipeline()

    def initializePipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device_name)

    def tokenizeInput(self, inputBatch):
        return self.tokenizer(inputBatch, padding=self._padding, truncation=self._truncation, max_length=self._max_length, return_tensors="pt").to(self.device_name)

    def encodeInput(self,tokenizedInput):
        return self.model(**tokenizedInput)

    def feature_extraction(self,inputBatch):
        tokenizedInput = self.tokenizeInput(inputBatch)
        return self.encodeInput(tokenizedInput)

if __name__ == "__main__":

    # set question directory location as variable
    currDir = os.getcwd()
    path_parent = os.chdir(os.path.dirname(os.getcwd()))
    print(os.getcwd())

    subset_dir = os.path.join(os.getcwd(),'multimodal-vqa','project_subset')
    train_subset_dir = os.path.join(subset_dir,'train_balanced_questions.csv')
    val_subset_dir = os.path.join(subset_dir,'val_balanced_questions.csv')
    test_subset_dir = os.path.join(subset_dir,'test_balanced_questions.csv')

    train_subset_imageID = []
    train_subset_questions = []
    train_subset_answers = []
    # train_subset_groundtruth_answers = []
    val_subset_imageID = []
    val_subset_questions = []
    val_subset_answers = []
    test_subset_questions = []

    # answer_distribution_top = {'yes': 1, 'no': 1,
    #                             'left': 2, 'right': 2,
    #                             'man': 3, 'woman': 3}
    # answer_distribution_list = ['yes\\no', 'left\\right', 'man\\woman']

    with open(train_subset_dir) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                train_subset_imageID.append(str(row[5])) # image ID
                train_subset_questions.append(str(row[4])) # question
                train_subset_answers.append(str(row[8])) # answer
            line_count += 1

    with open(val_subset_dir) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                val_subset_imageID.append(str(row[5])) # image ID
                val_subset_questions.append(str(row[4])) # question
                val_subset_answers.append(str(row[8])) # answer
            line_count += 1


    # with open(train_subset_dir) as csv_file:
    #     csv_reader = csv.reader(csv_file,delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         if line_count > 0 and str(row[8]) in answer_distribution_top.keys():
    #             train_subset_questions.append(str(row[4]))
    #             # train_subset_groundtruth_answers.append(str(row[8]))
    #             train_subset_answers_index = answer_distribution_top[str(row[8])]
    #             train_subset_answers.append(answer_distribution_list[train_subset_answers_index-1])
    #         line_count += 1

    # with open(val_subset_dir) as csv_file:
    #     csv_reader = csv.reader(csv_file,delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         if line_count > 0:
    #             val_subset_questions.append(str(row[4]))
    #         line_count += 1

    # with open(test_subset_dir) as csv_file:
    #     csv_reader = csv.reader(csv_file,delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         if line_count > 0:
    #             test_subset_questions.append(str(row[2]))
    #         line_count += 1

    # answer_distribution = {}
    # for answer in train_subset_answers:
    #     if answer in answer_distribution.keys():
    #         answer_distribution[answer] += 1
    #     else:
    #         answer_distribution[answer] = 1

    # sorted_answer_distribution = dict(sorted(answer_distribution.items(), key=lambda item: item[1], reverse=True))

    # models:
    # bert-base-uncased
    # roberta-base
    # deepset/sentence_bert

    device_name = "cuda:0"
    model_name = "roberta-base"
    baseBERTmodel = modelBERT(model_name, device_name)


    totDataSize = 1024
    batchSize = 32
    count = 0
    offsetVal = 0
    sizeKeys = len(train_subset_questions)
    totSizeReached = False

    totFeatureArr = torch.zeros(1,768)
    ansArr = []
    totQuestArr = []

    while(totSizeReached is False):
        print("While Loop Running: Count: " + str(count) + " out of " + str(totDataSize))
        questArr = []
        
        for idx in range(offsetVal, sizeKeys):
            if idx > offsetVal + batchSize - 1:
                break
            # question = raw_data[keys[count]]['question']
            question = train_subset_questions[count]
            answer = train_subset_answers[count]
            # print(question)
            questArr.append(question)
            totQuestArr.append(question)
            ansArr.append(answer)
            count+=1
            if count >= totDataSize or count >= sizeKeys-1:
                totSizeReached = True
                break

        offsetVal = count

        features_baseBERT = baseBERTmodel.feature_extraction(questArr)
        features_baseBERT = features_baseBERT[1] # (batch_size, 768)
        totFeatureArr = torch.cat((totFeatureArr,features_baseBERT.cpu().detach()), dim=0)


    totFeatureArr = totFeatureArr[1:,:]

    pca = PCA(n_components=50)
    question_rep_pca = pca.fit_transform(totFeatureArr)



    tsne_reps = TSNE(n_components=2, random_state=0, verbose=1).fit_transform(question_rep_pca)
    kmeans = KMeans(n_clusters=10)
    labels = kmeans.fit_predict(tsne_reps)
    centroids = kmeans.cluster_centers_

    df_points = pd.DataFrame(tsne_reps, columns=['x','y'])
    df_points["labels"] = labels
    df_centroids = pd.DataFrame(centroids, columns=['x','y'])
    sns.set_theme(style="white", palette=None)
    axs = sns.relplot(data=df_points, x = "x", y = "y", hue = "labels", palette = "colorblind")
    axs._legend.remove()

    plt.scatter(centroids[:,0], centroids[:,1], s = 40, color='k')
    plt.title('RoBERTa model T-SNE plot of question embeddings')
    plt.show()


    k = 6
    topk = []
    centroid_idxs = []
    for centre in centroids:
        distances = []
        for i, point in enumerate(tsne_reps):
            if point[0] == centre[0] and point[1] == centre[1]:
                centroid_idxs.append(i)
            distances.append((i, (point[0] - centre[0]) ** 2 + (point[1] - centre[1]) ** 2))
        distances.sort(key=lambda tup: tup[1])
        distances = distances[:k]
        topk.append(distances)

    # m = TSNE(learning_rate = 40)
    # tsne_features = m.fit_transform(totFeatureArr.numpy())
    # tsne_x = tsne_features[:,0]
    # tsne_y = tsne_features[:,1]

    # sns.scatterplot(x=tsne_x, y=tsne_y, hue=ansArr)
    # # plt.legend(answer_distribution_list)
    # plt.title('T-SNE diagram of ' + model_name + " representations on GQA training questions")
    # plt.show()

    with open("train_deepset-sentencebert_clusters.txt", "w+") as f:
        for i, val in enumerate(topk):
            f.write("Cluster "+ str(i + 1) + "\n")
            for idx, _ in val:
                f.write(totQuestArr[idx]+ "\t" + ansArr[idx]+"\n")

    # sample_str = []
    # for val in topk:
    #     for idx, _ in val:
    #         sample_str.append(totQuestArr[idx])
    #     break






    # initialize torch array of zeros of the hidden state size (1,1, 768)
    # for loop:
    # process every 32 unit batches of the input
    # store features in the torch array
    # run sklearn k-means clustering on the resulting model




    print("Program Finished.")


