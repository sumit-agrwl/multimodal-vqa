import pickle
import matplotlib.pyplot as plt
import torch

import sys
import os
import numpy as np

plt.rc('font', size=12)

def pickleRead(currdir, filepath):
    filepath_abs = os.path.join(currdir, filepath)
    with open(filepath_abs, 'rb') as inputfile:
        data = pickle.load(inputfile)

    return data


if __name__ == "__main__":
    path_parent = os.chdir(os.path.dirname(os.getcwd()))
    currDir = os.getcwd()
    print(currDir)

    titleLabels = ['Question Only', 'Scene Graph Labels Only', 'Image Only', 'Question and Image', 'Question and Scene Graph Labels', 'Question, Scene Graph Labels and Images']
    filenames = ['SimpleQAModel_QuestionOnly_roberta-base_01.pkl', 'SimpleQAModel_SceneGraphWordsOnly_roberta-base_01.pkl', 'SimpleQAModel_ImageOnly_resnet18_roberta-base_01.pkl', 'SimpleQAModel_QuestionandImage_resnet18_roberta-base_01.pkl', 'SimpleQAModel_QuestionPlusSceneGraphWords_roberta-base_01.pkl', 'SimpleQAModel_QuestionSceneGraphandImage_resnet18_roberta-base_01.pkl']


    # plot training accuracy plot
    for idx in range(len(titleLabels)-1):
        data = pickleRead(currDir, filenames[idx])
        x = data['train_batchArr']
        y = np.array(data['train_accArr'])*100.0
        label = titleLabels[idx]
        plt.plot(x,y,label = label)

    plt.title('Training Accuracy vs Number of Samples')
    plt.xlabel('Number of Samples [thousands]')
    plt.ylabel('Prediction Accuracy [%]')
    plt.grid()
    plt.legend()
    plt.show()

    # plot training loss plot
    for idx in range(len(titleLabels)-1):
        data = pickleRead(currDir, filenames[idx])
        x = data['train_batchArr']
        y = data['train_lossArr']
        label = titleLabels[idx]
        plt.plot(x,y,label = label)

    plt.title('Training Loss vs Number of Samples')
    plt.xlabel('Number of Samples [thousands]')
    plt.ylabel('Magnitude of Loss')
    plt.grid()
    plt.legend()
    plt.show()

    # plot validation accuracy plot
    for idx in range(len(titleLabels)-1):
        data = pickleRead(currDir, filenames[idx])
        x = data['val_batchArr']
        y = np.array(data['val_accArr'])*100.0
        label = titleLabels[idx]
        plt.plot(x,y,label = label)

    plt.title('Validation Accuracy vs Number of Samples')
    plt.xlabel('Number of Samples [thousands]')
    plt.ylabel('Prediction Accuracy [%]')    
    plt.grid()
    plt.legend()
    plt.show()

    # plot validation loss plot
    for idx in range(len(titleLabels)-1):
        data = pickleRead(currDir, filenames[idx])
        x = data['val_batchArr']
        y = data['val_lossArr']
        label = titleLabels[idx]
        plt.plot(x,y,label = label)

    plt.title('Validation Loss vs Number of Samples')
    plt.xlabel('Number of Samples [thousands]')
    plt.ylabel('Magnitude of Loss')
    plt.grid()
    plt.legend()
    plt.show()

    for idx in range(len(titleLabels)-2, len(titleLabels)):
        data = pickleRead(currDir, filenames[idx])
        x = data['val_batchArr']
        y = np.array(data['val_accArr'])*100.0
        label = titleLabels[idx]
        plt.plot(x,y,label = label)

    plt.title('Validation Accuracy vs Number of Samples')
    plt.xlabel('Number of Samples [thousands]')
    plt.ylabel('Prediction Accuracy [%]')    
    plt.grid()
    plt.legend()
    plt.show()
