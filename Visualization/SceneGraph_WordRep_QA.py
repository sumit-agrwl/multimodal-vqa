import sys
import os
import numpy as np

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


from transformers import pipeline
from transformers import AutoTokenizer, BertModel

import json
import csv
import torch
import pickle

import torch.nn as nn
import torch.optim

# TODO define model
# 1. scene graph representation
# 2. word representation
# 3. concatenation
# 4. one FC layer
# 5. softmax layer
# 6. with loss function
# TODO xavier initialize FC on init

class scenegraph_question_model(nn.Module):
    def __init__ (self, model_name, device_name, answerVocabSize, sceneGraphSize=1024):
        super().__init__()
        self.model_name = model_name
        self.device_name = device_name
        self._padding = True
        self._truncation = True
        self._max_length = 128+1

        # pretrained BERT model for feature representation ( 1 x 768)
        self.initializePipeline()
        self.wordVocabSize = self.tokenizer.vocab_size

        # fully connect layer for scene graphs
        self._sceneGraph_InputSize = self.wordVocabSize # look at tokensize of tokenizer for help
        self._sceneGraph_OutputSize = sceneGraphSize
        self.sceneGraphEmbedding = nn.Sequential(
            nn.Embedding(self.wordVocabSize+1, self._sceneGraph_OutputSize,padding_idx=self.wordVocabSize), 

            nn.MaxPool2d(kernel_size=(self._max_length,1), stride = 1)
        )
        

        # nonlinear activation after scene graph fully connect layer
        self.tanHActivation = nn.Tanh()

        # fully connected layer output
        self._answerVocabSize = answerVocabSize
        # self._concatenatedSize = self._sceneGraph_OutputSize + 768
        self._concatenatedSize = self._sceneGraph_OutputSize
        self.fullyconnected1 = nn.Linear(self._concatenatedSize, self._answerVocabSize)

    def forward(self, sceneGraphVect, questionStrVar):
        # # TODO encode questionVar
        # questionEmbedding = self.feature_extraction(questionStrVar)
        # questionEmbedding = questionEmbedding[1] # (batch_size, 768)

        # TODO encode sceneGraphVar
        # Tokenizer Vocab Siz
        sceneGraphEmbedding_unpadded = self.tokenizeInput(sceneGraphVect).data['input_ids'].to(self.device_name)
        sceneGraphEmbedding_padded = torch.ones(sceneGraphEmbedding_unpadded.shape[0],self._max_length)*self.wordVocabSize
        sceneGraphEmbedding_padded = sceneGraphEmbedding_padded.to(self.device_name)
        sceneGraphEmbedding_padded[:,:sceneGraphEmbedding_unpadded.shape[1]] = sceneGraphEmbedding_unpadded
        sceneGraphEmbedding_padded = self.sceneGraphEmbedding(sceneGraphEmbedding_padded.long())
        # sceneGraphEmbedding_padded = torch.sum(sceneGraphEmbedding_padded, dim=1)
        sceneGraphEmbedding_padded = torch.squeeze(sceneGraphEmbedding_padded, dim = 1)



        # # TODO concatenate the two representations
        # concateVect = torch.cat((sceneGraphEmbedding_padded, questionEmbedding), 1)
        # concateVect = questionEmbedding
        concateVect = sceneGraphEmbedding_padded

        # TODO pass concatenation through fully connected layer
        output = self.fullyconnected1(concateVect)

        return output


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
    print("Starting Code for Question 02")

    # open json files for questions and answers
    currDir = os.getcwd()
    path_parent = os.chdir(os.path.dirname(os.getcwd()))
    print(os.getcwd())

    subset_dir = os.path.join(os.getcwd(),'multimodal-vqa','project_subset')
    train_subset_dir = os.path.join(subset_dir,'train_balanced_questions.csv')
    val_subset_dir = os.path.join(subset_dir,'val_balanced_questions.csv')

    train_subset_imageID = []
    train_subset_questions = []
    train_subset_answers = []
    val_subset_imageID = []
    val_subset_questions = []
    val_subset_answers = []


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

    # open json files for scene graphs
    scenegraphs_subset_dir = os.path.join(os.getcwd(),'multimodal-vqa','sceneGraphs')
    train_scenegraphs_dir = os.path.join(scenegraphs_subset_dir,'train_sceneGraphs.json')
    val_scenegraphs_dir = os.path.join(scenegraphs_subset_dir,'val_sceneGraphs.json')

    with open(train_scenegraphs_dir) as json_file:
        scenegraph_train_raw_data = json.load(json_file)

    with open(val_scenegraphs_dir) as json_file:
        scenegraph_val_raw_data = json.load(json_file)

    # process through scene graphs to only get scene graphs that matter out
    scenegraph_train_data = {}
    scenegraph_val_data = {}
    for imageID in train_subset_imageID:
        sceneGraph = scenegraph_train_raw_data[imageID]
        tempStr = ""      
        for objectVar in sceneGraph['objects']:
            tempStr += str(sceneGraph['objects'][objectVar]['name']) + " "
        scenegraph_train_data[imageID] = tempStr

    for imageID in val_subset_imageID:
        sceneGraph = scenegraph_val_raw_data[imageID]
        tempStr = ""      
        for objectVar in sceneGraph['objects']:
            tempStr += str(sceneGraph['objects'][objectVar]['name']) + " "
        scenegraph_val_data[imageID] = tempStr
    

    # convert answer to one hot vectors
    answer_to_ID_map = {}
    answer_count = 0
    for answer in train_subset_answers:
        if answer not in answer_to_ID_map.keys():
            answer_to_ID_map[answer] = answer_count
            answer_count += 1
    
    for answer in val_subset_answers:
        if answer not in answer_to_ID_map.keys():
            answer_to_ID_map[answer] = answer_count
            answer_count += 1

    answer_count += 1
    # TODO map answer to one hot vector
    # answerTrainTruthArr = torch.zeros(len(train_subset_answers),answer_count)
    answerTrainTruthArr = torch.zeros(len(train_subset_answers), 1)
    count = 0
    for answer in train_subset_answers:
        # print("answer number: " + str(count))
        answerTrainTruthArr[count] = answer_to_ID_map[answer]
        count+=1
        # tempAnswerVect = torch.zeros(1,answer_count)
        # tempAnswerVect[0][answer_to_ID_map[answer]] = 1
        # answerTrainTruthArr = torch.cat((answerTrainTruthArr,tempAnswerVect),0)

    answerValTruthArr = torch.zeros(len(val_subset_answers), 1)
    count = 0
    for answer in val_subset_answers:
        answerValTruthArr[count] = answer_to_ID_map[answer]
        count+=1

    # answerTrainTruthArr = answerTrainTruthArr[1:,:]

    # initialize model
    model_name = 'roberta-base'
    device_name = 'cuda:0'
    sceneGraphSize = 1024
    model = scenegraph_question_model(model_name, device_name, answer_count).to(device_name)

    # create training loop
    # xavier initialize the model
    # nn.init.xavier_uniform_(model.sceneGraphEmbedding[0].weight[:-1])
    nn.init.xavier_uniform_(model.fullyconnected1.weight)

    # create optimizer
    # separate optimizers for word embeddings and softmax layer
    momentum = 0.9
    weight_decay = 1e-4
    word_lr = 0.8
    fc_lr = 0.01
    optimizer_word = torch.optim.SGD(model.sceneGraphEmbedding.parameters(), momentum = momentum, lr = word_lr, weight_decay = weight_decay)
    # optimizer_softmax = torch.optim.SGD(model.fullyconnected1.parameters(), momentum = momentum, lr = fc_lr, weight_decay = weight_decay)
    lossFunc = nn.CrossEntropyLoss().to(device_name)


    # TODO: call train loop w/ validation set every set number of batches
    num_epochs = 10
    totSizeReached = False
    totDataSize = len(train_subset_questions)
    val_totDataSize = len(val_subset_questions)
    batchSize = 64
    eval_time = 40000
    model.train()
    val_lossArr = []
    val_accArr = []
    val_batchArr = []
    train_lossArr = []
    train_accArr = []
    train_batchArr = []
    softmaxFunc = nn.Softmax().to(device_name)
    globalCount = 0
    for epoch in range(num_epochs):
        print("Begin training for epoch: " + str(epoch+1))
        totSizeReached = False
        offsetVal = 0
        count = 0
        valTime = False
        train_acc = 0
        while (totSizeReached is False):
            questArr = []
            # ansArr = []
            graphArr = []
            for idx in range(offsetVal, offsetVal+batchSize):
                if count >= totDataSize:
                    totSizeReached = True
                    break
                question = train_subset_questions[count]
                graph = scenegraph_train_data[train_subset_imageID[count]]
                
                questArr.append(question)
                graphArr.append(graph)
                ansArr = answerTrainTruthArr[offsetVal:offsetVal+batchSize].long().to(device_name)
                ansArr = torch.squeeze(ansArr, dim = 1)
                count += 1
                globalCount += 1
                if globalCount%eval_time == 0:
                    val_batchArr.append(globalCount/1000.0)
                    valTime = True
            offsetVal = count

            output = model(graphArr, questArr)

            predicted_answers = torch.argmax(softmaxFunc(output), dim = 1)
            train_acc += torch.sum(ansArr == predicted_answers)            

            optimizer_word.zero_grad()
            # optimizer_softmax.zero_grad()

            loss = lossFunc(output, ansArr)

            loss.backward()

            optimizer_word.step()
            # optimizer_softmax.step()

            if totSizeReached is True:
                train_batchArr.append(globalCount/1000.0)
                train_lossArr.append(loss)
                train_accArr.append(train_acc.data.item() / count)
                train_acc = 0


            if valTime is True:
                val_lossArr.append(loss.cpu().detach().data.item())
                val_acc = 0
                model.eval()
                val_totSizeReached = False
                val_offsetVal = 0
                val_count = 0
                # TODO build validation batch: question, answer, scenegraph words
                
                while val_totSizeReached is False:
                    val_questArr = []
                    val_graphArr = []
                    for idx in range(val_offsetVal, val_offsetVal+batchSize):
                        if val_count >= val_totDataSize:
                            val_totSizeReached = True
                            break
                        question = val_subset_questions[val_count]
                        graph = scenegraph_val_data[val_subset_imageID[val_count]]
                        
                        val_questArr.append(question)
                        val_graphArr.append(graph)
                        val_ansArr = answerValTruthArr[val_offsetVal:val_offsetVal+batchSize].long().to(device_name)
                        val_ansArr = torch.squeeze(val_ansArr, dim = 1)
                        val_count += 1
                        # print(val_count)
                    val_offsetVal = val_count
                    # TODO pass batch through model
                    if len(val_graphArr) == 0 or len(val_questArr) == 0:
                        break
                    output = model(val_graphArr, val_questArr)

                    # TODO compute and record loss
                    loss = lossFunc(output,val_ansArr)

                    # TODO compute and record accuracy
                    predicted_answers = torch.argmax(softmaxFunc(output), dim = 1)
                    val_acc += torch.sum(val_ansArr == predicted_answers)
                    print("Validation: --  Number: " + str(len(val_accArr)+1) + " Batch: " + str(val_offsetVal))
            
                val_accArr.append(val_acc.data.item()/val_totDataSize)
                
                valTime = False
                print("Validation Loss: " + str(val_lossArr[-1]) + " Accuracy: " + str(val_accArr[-1]))

            model.train()
            print("Training -- Epoch: " + str(epoch) + " | Batch: " + str(offsetVal))

    save_pkl_title = "SimpleQAModel_SceneGraphWordsOnly_roberta-base_01.pkl"
    pkl_dict = {}
    parameters = {}
    parameters['optimizer'] = 'SGD with Momentum'
    parameters['momentum'] = momentum
    parameters['weight_decay'] = weight_decay
    parameters['word_lr'] = word_lr
    pkl_dict['parameters'] = parameters
    pkl_dict['val_batchArr'] = val_batchArr
    pkl_dict['val_accArr'] = val_accArr
    pkl_dict['val_lossArr'] = val_lossArr
    pkl_dict['train_batchArr'] = train_batchArr
    pkl_dict['train_accArr'] = train_accArr
    pkl_dict['train_lossArr'] = train_lossArr  

    with open(save_pkl_title, "wb") as outputfile:
        pickle.dump(pkl_dict, outputfile)  

    # plt.plot(val_batchArr, val_accArr)
    # plt.title('Simple QA Validation Accuracy vs Number of Training Samples')
    # plt.xlabel('Number of Batches [Thousands]')
    # plt.ylabel('Validation Accuracy [%]')
    # plt.grid()
    # plt.show()

    # plt.plot(val_batchArr, val_lossArr)
    # plt.title('Simple QA Loss vs Number of Training Samples')
    # plt.xlabel('Number of Batches [Thousands]')
    # plt.ylabel('Magnitude of loss')
    # plt.grid()
    # plt.show()

    print("Done.")

    # TODO plot k-means cluster and see top 4-5 questions and see how they're performing