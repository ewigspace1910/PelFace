import os
import argparse
import yaml
import tqdm
import torch
import numpy as np
import sys
sys.path.append(os.getcwd())
from modules.evaluate import evaluate_model, load_model
from modules.models import Backbone
from modules.dataloader import get_DataLoader, ValidDataset
from torch.nn import DataParallel
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/arcface.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    parser.add_argument("--p", type=str, help="the saved file path")
    parser.add_argument("--cpu", action='store_true', help="using cpu not cuda")
    return parser.parse_args()

def verify(cfg, nworker=2, p=None):
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x])
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=nworker)

    if cfg['module'].find('se') < 0:
        backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir')
    lse: 
        backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir_se')
    backbone = load_model(backbone, p)
    backbone = backbone.to(device)
        
    backbone.eval()
    
    with torch.no_grad():
        for x in cfg['valid_data']:
            print("\nValidate...", x)
            accs, best_thresholds, eers = evaluate_model(backbone, valid_set[x], device=device, mode=cfg['mode'])
            #print('\t--{}\'s max accuracy list: {:.5f} \t best threshold: {} \teer: {:.5f}'.format(x,max(accs), thrs, eer))
            print("--accs: ", accs)
            print("--best_thresholds: ", best_thresholds)
            print("--eers: ", eers)

def main():
    pass

if __name__ == "__main__":
    args = get_args()
    if args.cpu:
        device =  torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        config['cpu'] = True
    else:
        # assert False, "unsupport run with gpu"
        device =  torch.device('cuda')
    with open(args.c, 'r') as file:
        print(args.c)
        config = yaml.load(file, Loader=yaml.Loader)
    verify(config, args.p)
