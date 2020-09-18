import glob
import pandas as pd
import time
from IPython import embed
import numpy as np
import os,sys,time
import matplotlib.pyplot as plt
from math import factorial
from tqdm import tqdm 
import pickle
from torch.utils.data import Dataset
import copy
import datetime

class PowerDataset(Dataset):
    def __init__(self, data, with_label=True):
        self.dataset = data
        self.with_label = with_label
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        if self.with_label:
            features = self.dataset[idx,:-1].astype(np.float32)
            label = self.dataset[idx,-1].astype(np.float32)
        else:    #no with label, must be real test set
            features = self.dataset[idx,:].astype(np.float32)
            label = 'Nop'
        sample = {"features": features, "label": label}
        return sample

def get_data(args):
    def _data_balance(datas, uniform_index):
        return
    quick_path = "../data/tmp_del/quick.pkl"
    #Quick or Normal?
    if os.path.exists(quick_path) and args.Quick_data:
        print("Reading from quick data: %s"%quick_path)
        datas = pickle.load(open(quick_path, 'rb'))
    else:
        print("Not quick")
        files = glob.glob("../data/气象*.csv")
        files.sort()                          
        features = pd.DataFrame()
        #features:
        for _file_ in tqdm(files[:]):
            feature = pd.read_csv(_file_, sep=',', low_memory=False, index_col=None)
            feature['时间'] = pd.to_datetime(feature['时间'],format='%Y/%m/%d %H:%M')
            feature = feature.set_index('时间')
            features = pd.concat((features, feature), axis=1)
        features['month'] = features.index.month
        features['day'] = features.index.day
        features['hour'] = features.index.hour
        features['minute'] = features.index.minute
        #labels:
        labels = pd.read_csv('../data/实测数据(装机容量10MW).csv', sep=',', low_memory=False, index_col=None)
        labels['Time'] = pd.to_datetime(labels['Time'],format='%Y/%m/%d %H:%M')
        labels = labels.set_index('Time')
        datas = pd.concat((features, labels), axis=1)
        print("Dumping to quick data: %s"%quick_path)
        pickle.dump(datas, open(quick_path, 'wb'), protocol=4)
    datas = datas.fillna(-1)   #fill 完之后，夜间的数据也会有标签
    labeled_data = datas[datas.index < datetime.datetime.fromisoformat('2019-01-01')]    #train: before 2019-01-01   test:2019-01-01  ~ 2020-01-01
    unlabeled_data = datas[datas.index >= datetime.datetime.fromisoformat('2019-01-01')]    #2019-01-01  ~ 2020-01-01
    unlabeled_data = unlabeled_data[unlabeled_data.index < datetime.datetime.fromisoformat('2020-01-01')]    #2019-01-01  ~ 2020-01-01
    return labeled_data, unlabeled_data

class normalizer(object):
    def __init__(self, X, Y, args):
        self.X_mean_never_touch_dynamic = X.mean(axis=0)
        self.X_std_never_touch_dynamic = X.std(axis=0)+1e-9
        self.Y_mean_never_touch_dynamic = np.array([Y.mean()])
        self.Y_std_never_touch_dynamic = np.array([Y.std()+1e-9])
        self.X = X
        self.Y = Y
    def generate_statistics(self):
        return
    def get_statistics(self, X_shape):
        return
    def normalize_X(self, X):
        return (X-self.X_mean_never_touch_dynamic)/self.X_std_never_touch_dynamic
    def normalize_XY(self, X, Y):
        return (X-self.X_mean_never_touch_dynamic)/self.X_std_never_touch_dynamic, (Y-self.Y_mean_never_touch_dynamic)/self.Y_std_never_touch_dynamic
    def denormalize_Y(self, Y):
        return Y*self.Y_std_never_touch_dynamic+self.Y_mean_never_touch_dynamic



def split_dataset(args, X, Y, raw_data):
    all_index = list(range(len(Y)))
    centers_around = [] 
    split_of_validation = 8 
    for i in range(1, split_of_validation+1): 
        centers_around.append(i/(split_of_validation+1))
    val_index = []
    for center_around in centers_around:
        val_index += list(range(int(len(Y)*(center_around-args.test_ratio/(2*split_of_validation))), int(len(Y)*(center_around+args.test_ratio/(2*split_of_validation)))))
    train_index = list(set(all_index) - set(val_index))
    print(len(train_index), 'for train,', len(val_index), 'for val')
    X_train = X[:, train_index]
    Y_train = Y[train_index]
    X_val = X[:, val_index]
    Y_val = Y[val_index]
    if args.VISUALIZATION:
        plt.scatter(np.array(range(len(Y)))[train_index][::10], Y_train[::10], s=0.5, label="train")
        plt.scatter(np.array(range(len(Y)))[val_index][::10], Y_val[::10], s=0.5, label="val")
        plt.xlabel("Sample points")
        plt.ylabel("Forces need to compensate")
        plt.title("Split of rain and val set")
        plt.legend()
        plt.show()
    raw_data_train = raw_data.iloc[train_index]
    raw_data_val = raw_data.iloc[val_index]
    return X_train, Y_train, X_val, Y_val, raw_data_train, raw_data_val


