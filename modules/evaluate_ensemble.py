from email.mime import image
from multiprocessing import pool
import os
import torch
import numpy as np
import time
import tqdm
import billiard as multiprocessing
from PIL import Image
from torchvision import transforms as T
import multiprocessing as mp

def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output

def calculate_accuracy(threshold, dists, actual_issame):
    NUMBER_OF_MODEL = 3

    if dists.shape[0] == NUMBER_OF_MODEL:
        predict_issame = np.sum(np.less(dists, threshold), axis=0) > (NUMBER_OF_MODEL/2)  # shape (3, 6000)

    else:
        predict_issame = np.less(dists, threshold) # return element-wise comparison of dists and thresthold

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    tn = np.sum(np.logical_and(np.logical_not(predict_issame),np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    
    if dists.shape[0] == NUMBER_OF_MODEL:
        acc = float(tp + tn) / dists[0].size  
    else:
        acc = float(tp + tn) / dists.size
    return tpr, fpr, acc


def calculate_roc(inputs):
    thresholds, dists, actual_issame = inputs
    tprs = np.zeros(thresholds.shape)
    fprs = np.zeros(thresholds.shape)
    accuracy = np.zeros(thresholds.shape)

    for i, thres in enumerate(thresholds):
        tprs[i], fprs[i], accuracy[i] = calculate_accuracy(thres, dists, actual_issame)
    
    best_thresholds = thresholds[np.argmax(accuracy)]
    #tpr = np.mean(tprs)
    #fpr = np.mean(fprs)
    return tprs, fprs, accuracy, best_thresholds


def calculate_eer(tprs, fprs):
    '''find a point to FNR = FPR'''
    fnrs = 1. - tprs
    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fprs[np.nanargmin(np.absolute((fnrs - fprs)))]
    eer_2 = fnrs[np.nanargmin(np.absolute((fnrs - fprs)))]
    return (eer_1 + eer_2) / 2 


def evaluate_model(model, dataset, device=torch.device('cpu')):
    dists = np.array([]) #distants
    labels = np.array([]) #labels
    for img1, img2, label in tqdm.tqdm(dataset):
        label = label.cpu().data.numpy() == 1
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        embds_1 = model(img1)
        embds_2 = model(img2)

        embds_1 = embds_1.cpu().data.numpy()
        embds_2 = embds_2.cpu().data.numpy()
        
        embds_1 = l2_norm(embds_1)
        embds_2 = l2_norm(embds_2)

        diff = np.subtract(embds_1, embds_2)
        dist = np.sum(np.square(diff), axis=1)

        labels = np.hstack((labels, label))
        dists  = np.hstack((dists, dist))
    
    thresholds = np.arange(0, 4, 0.01) 
    tprs, fprs, accs, best_thresholds = calculate_roc((thresholds, dists, labels))
    eer = calculate_eer(tprs, fprs)
    return accs, best_thresholds, eer

def ensem_evaluate_model(list_model, dataset, device=torch.device('cpu'), mode='hard'):
    '''
    Args
        mode : must be 'soft', 'hard', or 'wsoft', 'whard'
    Return 
        acc_array, best_threshold, err_value 
    '''
    dists = [np.array([])] * len(list_model) #distants
    labels = np.array([]) #labels

    n = len(list_model) # number of models 
    model_dists = np.array([])

    for img1, img2, label in tqdm.tqdm(dataset):
        label = label.cpu().data.numpy() == 1
        img1 = img1.to(device)
        img2 = img2.to(device)

        for i, model in enumerate(list_model):

            embds_1 = model(img1)
            embds_2 = model(img2)

            embds_1 = embds_1.cpu().data.numpy()
            embds_2 = embds_2.cpu().data.numpy()

            embds_1 = l2_norm(embds_1)
            embds_2 = l2_norm(embds_2)

            diff = np.subtract(embds_1, embds_2)
            dist = np.sum(np.square(diff), axis=1)

            dists[i] = np.hstack((dists[i], dist))

        labels = np.hstack((labels, label))

    dists = np.array(dists)

    # hard voting
    if mode == 'hard':

        accs, best_thresholds, eers = [], [], []
        thresholds = np.arange(0, 4, 0.01) 

        for i in range(0, n):
            tprs, fprs, acc, best_threshold = calculate_roc((thresholds, dists[i], labels))
            eer = calculate_eer(tprs, fprs)

            accs.append(max(acc))
            best_thresholds.append(best_threshold)
            eers.append(eer)

        tprs, fprs, acc, best_threshold = calculate_roc((thresholds, dists, labels))
        eer = calculate_eer(tprs, fprs)
        
        accs.append(max(acc))
        best_thresholds.append(best_threshold)
        eers.append(eer)

        return accs, best_thresholds, eers

    # soft voting
    if mode == 'soft':
        ensemble_dist = np.average(dists, axis=0) # Calculate distance of emsembled model 

        dists = np.vstack((dists, ensemble_dist)) # add emsemble_dist to the end of dists array

        thresholds = np.arange(0, 4, 0.01) 
        accs, best_thresholds, eers = [], [], []

        for i in range(0, n + 1):
            tprs, fprs, acc, best_threshold = calculate_roc((thresholds, dists[i], labels))
            eer = calculate_eer(tprs, fprs)

            accs.append(max(acc))
            best_thresholds.append(best_threshold)
            eers.append(eer)
        
        return accs, best_thresholds, eers

def simple_ensem_evaluate(list_model, dataset, device=torch.device('cpu'), mode='hard'):
    '''
    Args
        mode : must be 'soft', 'hard', or 'wsoft', 'whard'
    Return 
        acc_array, best_threshold, err_value 
    '''
    dists = [np.array([])] * len(list_model) #distants
    labels = np.array([]) #labels

    n = len(list_model) # number of models 
    model_dists = np.array([])

    for img1, img2, label in tqdm.tqdm(dataset):
        label = label.cpu().data.numpy() == 1
        img1 = img1.to(device)
        img2 = img2.to(device)

        for i, model in enumerate(list_model):

            embds_1 = model(img1)
            embds_2 = model(img2)

            embds_1 = embds_1.cpu().data.numpy()
            embds_2 = embds_2.cpu().data.numpy()

            embds_1 = l2_norm(embds_1)
            embds_2 = l2_norm(embds_2)

            diff = np.subtract(embds_1, embds_2)
            dist = np.sum(np.square(diff), axis=1)

            dists[i] = np.hstack((dists[i], dist))

        labels = np.hstack((labels, label))

    dists = np.array(dists)

    # hard voting
    if mode == 'hard':
        
        thresholds = np.arange(0, 4, 0.01) 
        tprs, fprs, acc, best_threshold = calculate_roc((thresholds, dists, labels))
        eer = calculate_eer(tprs, fprs)
        return max(acc), best_threshold, eer

    # soft voting
    if mode == 'soft':
        print("softvoting")
        ensemble_dist = np.average(dists, axis=0) # Calculate distance of emsembled model 
        thresholds = np.arange(0, 4, 0.01) 

        tprs, fprs, acc, best_threshold = calculate_roc((thresholds, ensemble_dist, labels))
        eer = calculate_eer(tprs, fprs)
        return max(acc), best_threshold, eer
        
device = torch.device('cpu')
def _infer(inputs):
    model, dataset = inputs
    dists =  np.array([])
    labels = np.array([])
    for img1, img2, label in tqdm.tqdm(dataset):
        label = label.cpu().data.numpy() == 1
     
        img1 = img1.to(device)
        img2 = img2.to(device)
    
        embds_1 , embds_2 = model(img1), model(img2)

        embds_1 = embds_1.cpu().data.numpy()
        embds_2 = embds_2.cpu().data.numpy()

        embds_1 = l2_norm(embds_1)
        embds_2 = l2_norm(embds_2)

        diff = np.subtract(embds_1, embds_2)
        dist = np.sum(np.square(diff), axis=1)

        dists = np.hstack((dists, dist))
        labels = np.hstack((labels, label))
    return dists, labels


def parallel_ensem_evaluate_model(dist_asyncs, device=torch.device('cpu'), mode='hard'):
    '''
    Args
        mode : must be 'soft', 'hard', or 'wsoft', 'whard'
    Return 
        acc_array, best_threshold, err_value 
    '''
    dists = [np.array([])] * 3 #distants
    labels = np.array([]) #labels
    print("start parallel")

    # dist_asyncs = list_model#.get()
    for i, (dist, label)  in enumerate(dist_asyncs):  
        dists[i] = dist
        labels = label
    dists = np.array(dists)

    # hard voting
    if mode == 'hard':
        thresholds = np.arange(0, 4, 0.01) 
        tprs, fprs, acc, best_threshold = calculate_roc((thresholds, dists, labels))
        eer = calculate_eer(tprs, fprs)

        return max(acc), best_threshold, eer

    # soft voting
    if mode == 'soft':
        ensemble_dist = np.average(dists, axis=0) # Calculate distance of emsembled model 
        thresholds = np.arange(0, 4, 0.01) 
        tprs, fprs, acc, best_threshold = calculate_roc((thresholds, ensemble_dist, labels))
        eer = calculate_eer(tprs, fprs)
        return max(acc), best_threshold, eer
        




def only_voting(dists, labels, device=torch.device('cpu'), mode='hard'):
    
   
    # hard voting
    if mode == 'hard':
        thresholds = np.arange(0, 4, 0.01) 
        tprs, fprs, acc, best_threshold = calculate_roc((thresholds, dists, labels))
        eer = calculate_eer(tprs, fprs)

        return max(acc), best_threshold, eer

    # soft voting
    if mode == 'soft':
        ensemble_dist = np.average(dists, axis=0) # Calculate distance of emsembled model 
        thresholds = np.arange(0, 4, 0.01) 
        tprs, fprs, acc, best_threshold = calculate_roc((thresholds, ensemble_dist, labels))
        eer = calculate_eer(tprs, fprs)
        return max(acc), best_threshold, eer

    #else
    print("calculate roc - 3")
    thresholds = np.arange(0, 4, 0.01) 
    tprs, fprs, acc, best_threshold = calculate_roc((thresholds, dists, labels))
    eer = calculate_eer(tprs, fprs)
    return max(acc), best_threshold, eer


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
