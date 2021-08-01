# -*- encoding: utf-8 -*-
'''
@File    :   generator.py
@Time    :   2021/07/28 11:31:29
@Author  :   Liu Qidong
@Version :   2.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import pickle
import random
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import torch
from torch.utils.data import Dataset, DataLoader


class RecData(Dataset):
    '''Input torch tensor for taking out data.'''
    def __init__(self, inter, user_feature, item_feature, uid, iid, mode='train'):
        super().__init__()
        self.inter = inter
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.uid_dict = uid
        self.iid_dict = iid
        self.mode = mode

    def __len__(self):

        return len(self.inter)

    def __getitem__(self, index):

        instance = self._merge_features(self.inter[index])
        if self.mode == 'train':
            return torch.LongTensor(instance[:-1]), torch.Tensor([instance[-1]])
        elif self.mode == 'test':
            return torch.LongTensor(instance)
        else:
            raise ValueError


    def _merge_features(self, inter):
        '''merge features into interaction data'''

        user_id = self.uid_dict[inter[0]]
        item_id = self.iid_dict[inter[1]]
        # 取训练集中的其他特征
        label = list(inter[2:])
        user_feature = list(self.user_feature[user_id])
        item_feature = list(self.item_feature[item_id])
        instance = user_feature + item_feature + label

        return np.array(instance)



class DataGenerator():
    '''Generate data for model.'''
    def __init__(self, args, user_feats=['user_id'], item_feats=['item_id'], train_feats=[], mode='offline'):

        self.mode = mode
        self.args = args
        self.data_path = './data/'
        self._load_data()
        self._map_features(user_feats, item_feats)
        self.features = user_feats + item_feats + train_feats
        
        
    def _load_data(self):
        '''Load data from pickle.'''
        if self.mode == 'offline':
            with open(self.data_path + 'train.pkl', 'rb') as f:
                self.train = pickle.load(f)
            with open(self.data_path + 'validation.pkl', 'rb') as f:
                self.test = pickle.load(f)
        elif self.mode == 'online':
            with open(self.data_path + 'train_online.pkl', 'rb') as f:
                self.train = pickle.load(f)
            with open(self.data_path + 'test.pkl', 'rb') as f:
                self.test = pickle.load(f)
        else:
            raise ValueError
        
        with open(self.data_path + 'user_feature.pkl', 'rb') as f:
            self.user_feature = pickle.load(f)
        with open(self.data_path + 'item_feature.pkl', 'rb') as f:
            self.item_feature = pickle.load(f)
        with open(self.data_path + 'id_dict.pkl', 'rb') as f:
            self.uid_dict, self.iid_dict = pickle.load(f)


    def _map_features(self, user_feats, item_feats):
        '''Get features that will be used in model'''
        self.user_feature = self.user_feature[user_feats]
        self.item_feature = self.item_feature[item_feats]
        self.user_feature = self.user_feature.to_numpy()
        self.item_feature = self.item_feature.to_numpy()

    
    def _merge_features(self, inter):
        '''merge features into interaction data'''
        print('Merging Features...')
        data = []
        for pair in tqdm(inter):
            user_id = self.uid_dict[pair[0]]
            item_id = self.iid_dict[pair[1]]
            # 取训练集中的其他特征
            label = list(pair[2:])
            user_feature = list(self.user_feature[user_id])
            item_feature = list(self.item_feature[item_id])
            instance = user_feature + item_feature + label
            data.append(instance)

        return data

    
    def make_train_loader(self):
        '''Make train dataloader'''
        print('Make train data...')

        trainset = RecData(self.train, self.user_feature, self.item_feature,
                           self.uid_dict, self.iid_dict)

        return DataLoader(trainset, 
                          batch_size=self.args.bs,
                          shuffle=True,
                          num_workers=self.args.num_workers,
                          drop_last=True,
                          collate_fn=lambda x: collate_point(x, self.features))


    def make_test_loader(self):
        '''Make test dataloader'''
        print('Make evaluation data...')

        if self.mode == 'offline':
            index = random.choices(list(range(len(self.test))), k=50000)
            testset = RecData(self.test[index], self.user_feature, self.item_feature,
                              self.uid_dict, self.iid_dict)
        elif self.mode == 'online':
            testset = RecData(self.test, self.user_feature, self.item_feature,
                              self.uid_dict, self.iid_dict, mode='test')
        # set the max batch size
        bs = min(testset.__len__(), 10000)

        return DataLoader(dataset=testset,
                          batch_size=bs,
                          num_workers=self.args.num_workers,
                          collate_fn=lambda x: collate_point(x, self.features, self.mode))


        
def collate_point(data, features=['user_id', 'item_id'], mode='offline'):
    '''
    Collate samples for point-wise method.
    Input: (bs, 2)-->(x, y)
    '''
    batch_data = {}

    if mode == 'offline':
        x = list(map(lambda x: x[0], data)) # take out features
    elif mode == 'online':
        x = data
    else:
        raise ValueError

    x = torch.stack(x)  # (bs, feat_num)
    for i, feat in enumerate(features):
        batch_data[feat] = x[:, i].long()

    if mode == 'offline':
        y = list(map(lambda x: x[1], data))
        y = torch.stack(y)  # (bs, 1)
        #y = y.unsqueeze(1)
        return batch_data, y

    elif mode == 'online':
        return batch_data
    

