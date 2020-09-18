#coding:utf-8
import sys
import time
from IPython import embed
import numpy as np
import os,sys,time
import warnings
warnings.filterwarnings("ignore")
import copy
import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import scipy
import argparse
from tqdm import tqdm

#my modules:
import data_stuff
import NN_model
import plot_utils
import evaluate

import pickle as pkl
import PIL
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(44)
import pandas as pd
import datetime

def train(args, model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    LOSS = 0
    pbar = tqdm(train_loader)
    for batch_idx, batch_samples in enumerate(pbar):
        data, target = batch_samples["features"], batch_samples["label"]
        data, target = data.to(device), target.to(device).reshape(-1, 1)
        optimizer.zero_grad()
        output = model((data))
        loss = F.mse_loss(output, target)
        loss.backward()
        LOSS += F.mse_loss(output, target, reduction='sum').item() # pile up all loss
        optimizer.step()
        if (batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)-1)and(batch_idx!=0):
            #pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
            pass
    train_loss_mean = LOSS/len(train_loader.dataset)
    return train_loss_mean, output, target

def validate(args, model, device, validate_loader):
    model = model.to(device)
    model.eval()
    LOSS = 0
    with torch.no_grad():
        pbar = tqdm(validate_loader)
        for batch_idx, batch_samples in enumerate(pbar):
            data, target = batch_samples["features"], batch_samples["label"]
            data, target = data.to(device), target.to(device).reshape(-1, 1)
            output = model((data))
            LOSS += F.mse_loss(output, target, reduction='sum').item() # pile up all loss
            #pbar.set_description('Validate: [{}/{} ({:.0f}%)]'.format(idx*len(data), len(validate_loader.dataset), 100.*idx/len(validate_loader)))
    validate_loss_mean = LOSS/len(validate_loader.dataset)
    return validate_loss_mean, output, target

def test(args, model, device, test_loader):
    model = model.to(device)
    model.eval()
    prediction = []
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, batch_samples in enumerate(pbar):
            data = batch_samples["features"]
            data = data.to(device)
            output = model((data))
            prediction = np.append(prediction, np.array(output.cpu()))
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--learning_rate', '-LR', type=float, default=5e-2)
    parser.add_argument('--test_ratio', '-TR', type=float, default=0.2)
    parser.add_argument('--max_epoch', '-E', type=int, default=1000)

    parser.add_argument('--hidden_width_scaler', type=int, default = 5)
    parser.add_argument('--hidden_depth', type=int, default = 5)
    parser.add_argument('--Cuda_number', type=int, default = 1)
    parser.add_argument('--num_of_batch', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    parser.add_argument('--NO_CUDA', action='store_true', default=False)
    parser.add_argument('--Quick_data', "-Q", action='store_true', default=False)
    parser.add_argument('--mode', type=str, choices=["R", "P"], default='P')  #直接预测P还是先预测R
    args = parser.parse_args()
    if not args.NO_CUDA:
        device = torch.device("cuda", args.Cuda_number)
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    #-------------------------------------Do NN--------------------------------:
    #Get data:
    raw_labeled_data, raw_unlabeled_data = data_stuff.get_data(args)
    data_X = raw_labeled_data.values[:,:-2]
    data_full_X = raw_unlabeled_data.values[:,:-2]  
    input_columns_names = raw_labeled_data.columns[:-2]
    if args.mode == 'R':
        data_Y = raw_labeled_data.values[:,-2].reshape(-1, 1)
        output_columns_names = raw_labeled_data.columns[-2]
    elif args.mode == 'P':
        data_Y = raw_labeled_data.values[:,-1].reshape(-1, 1)
        output_columns_names = raw_labeled_data.columns[-1]
    else:
        print("Error mode!")
        sys.exit()
    print("Shape of all input: %s, shape of all output: %s"%(data_X.shape, data_Y.shape))

    #Normalize data:
    normer = data_stuff.normalizer(data_X, data_Y, args)
    #X_normed, Y_normed = normer.normalize_XY(data_X, data_Y)
    X_normed = normer.normalize_X(data_X)
    X_full_normed = normer.normalize_X(data_full_X)
    Y_normed = data_Y   #No, no nomalize output data

    #MUST CHECK INPUT DISTRIBUTION!!!!!!
    if args.VISUALIZATION and False:
        print("Checking distribution of all data...")
        plot_utils.check_distribution_power(X_normed, input_columns_names, args)
        plot_utils.check_distribution_power(Y_normed, output_columns_names, args)
    #TODO: skewed distribution ????? how to convert
    #TODO: compressing features?

    #形成Dataset：
    raw_data_normed = np.hstack((X_normed, Y_normed))
    raw_full_data_normed = X_full_normed
    power_dataset = data_stuff.PowerDataset(raw_data_normed)
    power_test_dataset = data_stuff.PowerDataset(raw_full_data_normed, with_label=False)
    #划分数据集：
    train_size = int((1-args.test_ratio) * len(power_dataset))
    validate_size = len(power_dataset) - train_size
    test_size = len(power_test_dataset)
    train_dataset, validate_dataset = Data.random_split(power_dataset, [train_size, validate_size])
    test_dataset, _ = Data.random_split(power_test_dataset, [test_size, 0])
    #分别给loader （默认sampler）：
    _batch_size = int(len(train_dataset)/args.num_of_batch)
    train_loader = Data.DataLoader( 
            dataset=train_dataset, 
            batch_size=_batch_size,
            shuffle=False,
            drop_last=False,
	          num_workers=4,
            #pin_memory=True,
            )
    validate_loader = Data.DataLoader( 
            dataset=validate_dataset, 
            batch_size=validate_size,
            shuffle=False,
            drop_last=False,
	          num_workers=4,
            #pin_memory=True,
            )
    test_loader = Data.DataLoader( 
            dataset=test_dataset, 
            batch_size=test_size,    #full batch test,
            shuffle=False,
            drop_last=False,
	          num_workers=1,
            #pin_memory=True,
            )


    #Model parameters:
    input_size = X_normed.shape[1] 
    hidden_size = X_normed.shape[1]*args.hidden_width_scaler
    hidden_depth = args.hidden_depth
    output_size = Y_normed.shape[1]
    model = NN_model.NeuralNetSimple(input_size, hidden_size, hidden_depth, output_size, device)
    #model.load_state_dict(torch.load(args.restart_model_path).state_dict())
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.1)

    #训练：
    print("Now training...")
    plt.ion()
    train_loss_history = []
    validate_loss_history = []
    train_error_history = []
    validate_error_history = []
    history_error_ratio_val = []
    overall_error_history = []
    #This is for save gif, always on:
    if args.VISUALIZATION:
        plt.figure(figsize=(14, 8))
    best_model = None
    for epoch in range(int(args.max_epoch+1)):
        print("Epoch: %s"%epoch, "TEST AND SAVE FIRST")
        #Test first:
        #model.eval()
        #validate_loss, validate_outputs, validate_targets = validate(args, model, device, validate_loader)
        if epoch>=200:
            #if overall_error_history[-1] < np.array(overall_error_history[:-1]).min():
            if validate_loss_history[-1] < np.array(validate_loss_history[:-1]).min():
                #torch.save(model.eval(), "../outputs/NN_weights_best.pth")
                best_model = copy.deepcopy(model)
                print("***MODEL SAVED***")
        #Train/Val then:
        train_loss, train_outputs, train_targets = train(args, model, device, train_loader, optimizer, epoch)
        validate_loss, validate_outputs, validate_targets = validate(args, model, device, validate_loader)
        #Always save figure:
        #if args.VISUALIZATION:
        #    plot_utils.visual_predict_val(validate_targets.cpu(), validate_outputs.cpu(), epoch=epoch)
        #Infos:
        train_loss_history.append(train_loss)
        validate_loss_history.append(validate_loss)
        print("Train set  Average loss: {:.8f}".format(train_loss))
        print('Validate set Average loss: {:.8f}'.format(validate_loss))
    if args.VISUALIZATION:
        plt.close()
        plt.ioff()
        plt.clf()
        plt.plot(train_loss_history, label='train loss')
        plt.plot(validate_loss_history, label = 'val loss')
        plt.title("Train/Val loss history")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        #plt.draw()
        #plt.pause(2)
        plt.show()
        plt.close()
    names_note = "NN weights"
    print("Names note:", names_note)
    print("NN:", "NONE")

    if best_model is not None:
        prediction = best_model(torch.FloatTensor(X_full_normed).to(device)).cpu().detach().numpy()
    else:
        prediction = model(torch.FloatTensor(X_full_normed).to(device)).cpu().detach().numpy()
        print("Currently no best model")
    prediction = np.clip(prediction, min(Y_normed), max(Y_normed))
    np.savetxt('../outputs/prediction.txt', prediction)

    prediction = pd.DataFrame(prediction)
    prediction.index = raw_unlabeled_data.index
    prediction.to_csv('../outputs/prediction.csv', header=None)
    
    embed()

    #if not args.finetune:
    #    torch.save(model.eval(), "../outputs/NN_weights_%s_%s"%(args.further_mode, args.axis_num))
    #else:
    #    torch.save(model.eval(), "../outputs/NN_weights_%s_%s_finetune"%(args.further_mode, args.axis_num))
    #embed()


