
from __future__ import print_function, absolute_import
from cProfile import label
from dis import dis
from glob import glob
from math import dist
import os
import sys
sys.path.append(os.getcwd())
import argparse
import yaml
import time
import torch
import tqdm
import numpy as np
from modules.evaluate_ensemble import  imple_ensem_evaluate, parallel_ensem_evaluate_model, only_voting, _infer
from modules.evaluate import evaluate_model, load_model
from modules.models import Backbone
from modules.dataloader import get_DataLoader, ValidDataset
from torchvision import transforms as T
import multiprocessing
from PIL import Image
from multiprocessing import Process, Manager

device = torch.device("cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/ensemble/soft.yaml', help='config path')
    parser.add_argument('--b', type=int, default=16, help='batchsize')
    parser.add_argument('--l', type=int, default=5, help='number loop')
    parser.add_argument("--e", action='store_true', help="using ensemble or single model")
    parser.add_argument("--parallel", action='store_true', help="using parallel computation")
    parser.add_argument("--cpu", action='store_true', help="using cpu not cuda")

    return parser.parse_args()


def single_verify(cfg):
    print("single verify")
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x], only_path=False)
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=2)
    models = []

    for model_path in cfg['weight_path']:
        print("\t",model_path, end="\t")
        if cfg['module'][model_path].find('se') < 0:
            backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir')
        else: 
            backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir_se')
        if not cfg['cpu']:
            backbone = load_model(backbone, cfg['weight_path'][model_path])
        backbone = backbone.to(device)
        
        backbone.eval()

        models.append(backbone)
        break
    costs = {}
    with torch.no_grad():
        for x in cfg['valid_data']:
            dataset = [x for x in valid_set[x]]
            tstart = time.time()
            print("\nValidate...", x)
            dists = np.array([]) #distants
            labels = np.array([]) #labels
            
            for img1, img2, label in tqdm.tqdm(dataset):
                label = label.cpu().data.numpy() == 1
                img1 = img1.to(device)
                img2 = img2.to(device)
                
                embds_1 = models[0](img1)
                embds_2 = models[0](img2)

                embds_1 = embds_1.cpu().data.numpy()
                embds_2 = embds_2.cpu().data.numpy()
                

                diff = np.subtract(embds_1, embds_2)
                dist = np.sum(np.square(diff), axis=1)

                labels = np.hstack((labels, label))
                dists  = np.hstack((dists, dist))


            accs, best_thresholds, eers = only_voting(dists, labels, mode="single")
            # print('\t--{}\'s max accuracy list: {:.5f} \t best threshold: {} \teer: {:.5f}'.format(x,max(accs), thrs, eer))
            
            elapsed = time.time() - tstart
            # print("--accs: ", accs)
            # print("--best_thresholds: ", best_thresholds)
            # print("--eers: ", eers)
            costs[x] = [elapsed]
    return costs



def seq_ensemble_verify(cfg, nworker=1):
    print("consequently verify by ensemble")
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x], only_path=False)
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=nworker)
    models = []

    for model_path in cfg['weight_path']:
        print("\t",model_path, end="\t")
        if cfg['module'][model_path].find('se') < 0:
            backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir')
        else: 
            backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir_se')
        if not cfg['cpu']:
            backbone = load_model(backbone, cfg['weight_path'][model_path])
        backbone = backbone.to(device)
        
        backbone.eval()

        models.append(backbone)
    costs = {}
    with torch.no_grad():
        for x in cfg['valid_data']:
            dataset = [x for x in valid_set[x]]
            tstart  = time.time()
            print("\nValidate...", x)
            endists = [np.array([])] * len(models) #distants
            labels = np.array([]) #labels

            for img1, img2, label in tqdm.tqdm(dataset):
                label = label.cpu().data.numpy() == 1
                img1 = img1.to(device)
                img2 = img2.to(device)

                for j, model in enumerate(models):

                    embds_1 = model(img1)
                    embds_2 = model(img2)

                    embds_1 = embds_1.cpu().data.numpy()
                    embds_2 = embds_2.cpu().data.numpy()

                    diff = np.subtract(embds_1, embds_2)
                    dist = np.sum(np.square(diff), axis=1)

                    endists[j] = np.hstack((endists[j], dist))

                labels = np.hstack((labels, label))
            endists = np.array(endists)

            accs, best_thresholds, eers = only_voting(endists, labels, mode=cfg['mode'])            
            elapsed = time.time() - tstart
            # print("--accs: ", accs)
            # print("--best_thresholds: ", best_thresholds)
            # print("--eers: ", eers)
            costs[x] = [elapsed]
    return costs


MODELS = []
manager = Manager()
dists = manager.dict()
DS = manager.list() 

def pinfer(input):
    i = input
    model = MODELS[i]
    dist = np.array([])
    for img1, img2, _ in tqdm.tqdm(DS):
        embd_1 = model(img1.to(device)).data.numpy()
        embd_2 = model(img2.to(device)).data.numpy()
        diff = np.sum(np.subtract(embd_1, embd_2) ** 2, axis=1)
        dist = np.hstack((dist, diff))
    dists[i] = dist

    
def parallel_ensemble_verify(cfg, nworker=2):
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x], only_path=False)
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=nworker)
    models = []

    for model_path in cfg['weight_path']:
        print("\t",model_path, end="\t")
        if cfg['module'][model_path].find('se') < 0:
            backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir')
        else: 
            backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir_se')
        if not cfg['cpu']:
            backbone = load_model(backbone, cfg['weight_path'][model_path])
        backbone = backbone.to(device)
        backbone.eval()
        models.append(backbone)
    
    costs = {}
    with torch.no_grad():
        for x in cfg['valid_data']:
            global dists
            print("\nValidate...", x)
            global MODELS, DS
            MODELS = models
            if len(DS) == 0:
                DS = [x for x in valid_set[x]]
            print(len(DS))
            dists[0] = np.array([]) #distants
            dists[1] = np.array([]) #distants
            dists[2] = np.array([]) #distants
            labels = np.array([], dtype=np.int32) #labels
            for _, _, l in tqdm.tqdm(DS): labels = np.hstack((labels, l)) 
            print(labels.shape)
            ########################
            try: multiprocessing.set_start_method('spawn')
            except: pass
            pool = multiprocessing.Pool()
            tstart  = time.time()

            pool.map(pinfer, [m for m in range(3)])
            # pool.map_async(pinfer, [m for m in range(3)]).get()
            
            en_dist = np.array([dists[0], dists[1], dists[2]])
            accs, best_thresholds, eers = only_voting(en_dist, labels, device=device, mode=cfg['mode'])
            elapsed = time.time() - tstart
            #########################
            # print("--accs: ", accs)
            # print("--best_thresholds: ", best_thresholds)
            # print("--eers: ", eers)
            costs[x] = [elapsed]
    return costs



#==============================

def main():
    pass

if __name__ == "__main__":
    args = get_args()

    time_costs = {}
    with open(args.c, 'r') as file:
        print(args.c)
        config = yaml.load(file, Loader=yaml.Loader)
        config['batch_size'] = args.b
    if args.cpu:
        device =  torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        config['cpu'] = True
    else:
        # assert False, "unsupport run with gpu"
        device =  torch.device('cuda')
        
        config['cpu'] = False
    costs = {}
    nloop = args.l
    for i in range(nloop): 
        
        if not args.e : 
            cost = single_verify(config)
        elif args.parallel: 
            cost = parallel_ensemble_verify(config)
        else: 
            cost = seq_ensemble_verify(config)
        
        #save statistic
        print(cost)
        if i == 0: costs = cost
        else: 
            for x in cost: costs[x]+=cost[x]
        time.sleep(15)
    #print
    print(costs)
    for x in costs:
        if not args.e : 
            print("monitoring single verification runtime")
        elif args.parallel: 
            print("monitoring parallel verification runtime")
        else: 
            print("monitoring sequencently verification runtime")
        print("Total of time for infering on {}: {}s".format(x, sum(costs[x])/nloop))

