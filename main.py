# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/11/05 15:38:35
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import json
from models.layers.input import *
from models.deepfm import DeepFM
from generators.generator import DataGenerator

import os
import time
import argparse
import setproctitle
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Input args')

'''The arguments about model and training'''
parser.add_argument('-m', default='fm', 
                    choices=['fm', 'deepfm', 'mf'], help='choose model')
parser.add_argument('-dataset', default='ML1M', 
                    choices=['ML1M', 'Amazon', 'ML20M'], 
                    help='choose dataset')
parser.add_argument('-lr', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('-epoch', default=10, type=int, 
                    help='the number of epoch')
parser.add_argument('-bs', default=128, type=int, help='batch size')
parser.add_argument('-lr-decay', default=0.97, type=float, 
                    help='learning rate decay')
parser.add_argument('-lr-type', default='none', 
                    choices=['exp', 'step', 'cos', 'none'], 
                    help='method of learning rate decay')
parser.add_argument('-period', default=100, type=int, 
                    help='how many epochs to conduct a learning rate decay')
parser.add_argument('-es', default=False, action='store_true', 
                    help='early stop or not')
parser.add_argument('-optimizer', default='adam', 
                    choices=['sgd', 'adam', 'rmsprop'], 
                    help='choose the optimization of training')
parser.add_argument('-bn', default=0, type=bool, 
                    help='batch normalization or not')
parser.add_argument('-init', default='normal', 
                    choices=['normal', 'kaiming'], 
                    help='how initialize the neural network')
parser.add_argument('-train-test', default=0.9, type=float, 
                    help='the ratio of training size in whole dataset')
parser.add_argument('-train-val', default=0.9, type=float, 
                    help='the ratio of training size in training dataset')
parser.add_argument('-scale', default=1, type=int, 
                    help='wheather scale the x of dataset')
parser.add_argument('-load-m', default=False, action='store_true',
                    help='wheather load model rather than training.')
parser.add_argument('-bpr', default=False, action='store_true',
                    help='whether use bpr loss')

'''The arguments about log'''
parser.add_argument('-log', default=False, action='store_true', 
                    help='whether log the experiment.')
parser.add_argument('-log-path', default='default', 
                    choices=['default', 'time'], 
                    help='where is the log')
parser.add_argument('-batch-record', default=1000, type=int, 
                    help='output record once pre <batch-record> batches')


'''The arguments about device'''
parser.add_argument('-num-workers', default=0, type=int, 
                    help='equals to the number of cpu cores')
parser.add_argument('-use-cuda', default=False, action='store_true', 
                    help='whether use cuda')
parser.add_argument('-device-tab', default=0, type=int, 
                    help="specify the device number. note how many gpus you have.")
parser.add_argument('-task', type=str, default='classification',
                    choices=['classification', 'regression'], 
                    help='the task of output layer')

'''The arguments about specified model'''
parser.add_argument('-em-dim', default=16, type=int,
                    help='the dimension of embedding')
parser.add_argument('-alpha', default=0.1, type=float,
                    help='hyper-parameter')
parser.add_argument('-beta', default=0.1, type=float,
                    help='hyper-parameter')
parser.add_argument('-gamma', default=0.01, type=float,
                    help='hyper-parameter')
parser.add_argument('-loss-decay', default=0.9, type=float, 
                    help='the decay of auxiliary task weight')
parser.add_argument('-margin-decay', default=0.9, type=float,
                    help='the decay of sample margin')
parser.add_argument('-episilon', default=0.00015, type=float,
                    help='the hyperparameter for label smoothing')





def main(args, mode='offline'):

    with open('./data/info.json', 'r') as f:
        info = json.load(f)

    '''Step 1: Create item feat and user feat and feature list'''
    user_feats = ['user_id', 'user_device', 'user_system', 'user_province', 'user_city']
    item_feats = ['item_id', 'item_picture', 'item_cluster1']
    train_feats = ['network', 'refresh']
    feat_list = []
    for feat in user_feats + item_feats + train_feats:
        feat_list.append(sparseFeat(feat, info['vocabulary_size'][feat], args.em_dim))

    '''Step 2: Data generator'''
    data_generator = DataGenerator(args, user_feats, item_feats, train_feats, mode)

    '''Step 3: construct model and use cuda'''
    model = DeepFM(args, feat_list, data_generator)

    if args.use_cuda:
        model.to('cuda:' + str(args.device_tab))

    '''Step 3: train model or load model'''
    model.fit(mode)

    if mode == 'online':
        model_path = './save_model/' + args.m + '.ckpt'
        if os.path.exists(model_path):
            model = DeepFM(args, feat_list, data_generator)
            model.load_state_dict(torch.load(model_path))
            if args.use_cuda:
                model.to('cuda:' + str(args.device_tab))
        else:
            model._save_model()

        with torch.no_grad():
            submit(data_generator, model)

    print('Mission Complete!')


def submit(DG, model):

    loader = DG.make_test_loader()
    y = model._move_device(torch.Tensor())
    for batch in tqdm(loader):
        x = model._move_device(batch)
        batch_y = model(x)
        y = torch.cat([y, batch_y], dim=0)
    
    y = y.squeeze().cpu().detach().numpy()
    df = pd.DataFrame({'id': list(range(1, 50001)), 'label': y})
    now_str = time.strftime("%m%d%H%M", time.localtime())
    df.to_csv('./submission/' + now_str + '.csv', index=False, header=False)
    



if __name__ == '__main__':

    setproctitle.setproctitle("Qidong's Competition")
    args = parser.parse_args()
    main(args, mode='offline')
