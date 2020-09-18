#coding: utf-8
#python NN_deploy.py  -V --mode='low_high'
import sys
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import data_stuff
from IPython import embed
import evaluate
import NN_model
import warnings
import time
import copy
import plot_utils
warnings.filterwarnings("ignore")

def judge_features_secure(args, raw_data_X, raw_secure_range, input_columns_names):
    raw_data_range = np.array([np.abs(raw_data_X).min(axis=1), np.abs(raw_data_X).max(axis=1)])
    safe_matrix = np.array([raw_data_range[0,:]>=raw_secure_range[0,:], raw_data_range[1,:]<=raw_secure_range[1,:]])
    if safe_matrix.all() == True:
        return
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!Axis:%s FEATURE WARNING!!!!!!!!!!!!!!!!!!!!!!!\nThe training data not overlapping testing!"%args.axis_num)
        print(input_columns_names)
        print("Features Testing on:\n", raw_data_range, "\nFeatures Safe range:\n", raw_secure_range, "\nFeatures secure matrix:\n", safe_matrix)

def get_part_model(args, shape_X, name, axis_num):
    if not args.finetune:
        model_path = "../models/NN_weights_best_%s_%s"%(name, axis_num) if args.model_path is None else args.model_path
    else:
        model_path = "../models/NN_weights_best_%s_%s_finetune"%(name, axis_num) if args.model_path is None else args.model_path
    print("Loading part model:%s"%model_path)
    #model = torch.load(model_path, map_location=torch.device(args.device_type))
    model = NN_model.NeuralNetSimple(input_size=shape_X, hidden_size=shape_X*5, hidden_depth=3, output_size=1, device=torch.device(args.device_type))
    model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device_type)).state_dict())
    model = model.to(args.device_type)
    return model

def to_C(args, model_part1, model_part2, inputs):
    if 'all' not in args.mode:
        name1 = args.mode.split('_')[0]     #'acc'
        #name2 = args.mode.split('_')[1]     #'uniform'
    else:   # 'acc_uniform_all', 'low_high_all'
        name1 = args.mode.split('_')[-1]    #'all'
        #name2 = args.mode.split('_')[-1]    #'all' as well
    #Trace with jit:
    model_part1.eval()
    #model_part2.eval()
    traced_module1 = torch.jit.trace(model_part1.cpu(), inputs.cpu())
    #traced_module2 = torch.jit.trace(model_part2.cpu(), inputs.cpu())
    model1_path = "../models/NN_weights_%s_C_%s.pt"%(name1, args.axis_num)
    #model2_path = "../models/NN_weights_%s_C_%s.pt"%(name2, args.axis_num)
    traced_module1.save(model1_path)
    #traced_module2.save(model2_path)
    #print("C models saved", model1_path, model2_path)
    print("C models saved:", model1_path)

def get_data_one(args, raw_plan, mode):
    #Take out what we need:
    #Make inputs:
    raw_data_X = np.empty(shape=(0, len(raw_plan)))
    input_columns_names = []
    #data are index from 0, thus
    local_axis_num = args.axis_num - 1
    input_columns_names += ['axc_pos_%s'%local_axis_num]
    input_columns_names += ['axc_speed_%s'%local_axis_num]
    input_columns_names += ['axc_torque_ffw_gravity_%s'%local_axis_num]
    input_columns_names += ['axc_torque_ffw_%s'%local_axis_num]
    input_columns_names += ['Temp']
    raw_data_X = raw_plan[input_columns_names].values.T
    raw_data_Y = raw_plan['need_to_compensate'].values

    #Normalize data:
    normer = data_stuff.normalizer(raw_data_X, raw_data_Y, args)
    normer.get_statistics(raw_data_X.shape[1])
    raw_secure_range = normer.get_raw_secure()
    judge_features_secure(args, raw_data_X, raw_secure_range, input_columns_names)
    normed_data_X, normed_data_Y = normer.normalize_XY(raw_data_X, raw_data_Y)
    return normed_data_X, normed_data_Y, normer

def get_model_one(args, shape_X):
    #Get model:
    assert 'all' in args.mode, "I assume this is new version which train acc uniform together as all"
    model = get_part_model(args, shape_X, args.mode.split('_')[-1], args.axis_num)    #'all'
    return model, model

def get_compensate_force(args, normed_data_X, normed_data_Y, model_part1, model_part2, part1_index, part2_index):
    #Forward to get output:
    inputs = torch.FloatTensor(normed_data_X.T).to(args.device_type)
    output_part1 = model_part1(inputs[part1_index]).detach().cpu().numpy()
    output_part2 = model_part2(inputs[part2_index]).detach().cpu().numpy()
    #Speed test:
    #if args.speed_test:
    #    inputs[:,:]=1
    #    #On default device:
    #    for i in range(10): 
    #        model_part1(inputs[:args.buffer_length])
    #    tic = time.time() 
    #    for i in range(1000): 
    #        model_part1(inputs[:args.buffer_length])
    #    print((time.time()-tic)/1000*1000, "ms on %s"%device_type)  
    #    #On cpu deivce:
    #    cpu_model = copy.deepcopy(model_part1).cpu()
    #    cpu_inputs = inputs.cpu()
    #    for i in range(10): 
    #        cpu_model(cpu_inputs[:args.buffer_length])
    #    tic = time.time() 
    #    for i in range(1000): 
    #        cpu_model(cpu_inputs[:args.buffer_length])
    #    print((time.time()-tic)/1000*1000, "ms on cpu")  

    #import my_analyzer
    #my_analyzer.performance_shape(raw_plan, inputs, model_part1)
    #Compose together:
    #TODO: solve the switch point.
    output_full_series = np.zeros(normed_data_X.shape[1])
    output_full_series[part1_index] = output_part1.reshape(-1)
    output_full_series[part2_index] = output_part2.reshape(-1)

    #Save for Production:
    to_C(args, model_part1.cpu(), model_part2.cpu(), inputs.cpu())
    model_returned = model_part1
    return output_full_series, inputs


if __name__ == "__main__":
    print("NN Deploy test...")

    #Configs:
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--max_force', type=int, default = 1)
    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--buffer_length', type=int, default = 6)
    parser.add_argument('--mode', type=str, choices=["acc_uniform", "low_high", "acc_uniform_all", "low_high_all"], required=True)
    parser.add_argument('--data_path', type=str, default = "../data/planning.csv")
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--finetune', action='store_true', default=False)
    args = parser.parse_args()
    args.rated_torque = [5.7, 5.7, 1.02, 0.318, 0.318, 0.143][args.axis_num-1]
    cuda_is_available = torch.cuda.is_available()
    if args.no_cuda:
        cuda_is_available = False
    if not cuda_is_available:
        args.device_type = 'cpu'
        print("Using cpu")
    else:
        args.device_type = 'cuda'
        print("Using gpu")
    mode = ['deploy', args.mode]

    #Get planned data:
    raw_plan, part1_index, part2_index = data_stuff.get_data(args, mode)
    normed_data_X, normed_data_Y, normer = get_data_one(args, raw_plan, mode)

    #Get models:
    model_part1, model_part2 = get_model_one(args, normed_data_X.shape[0])

    #Data_preprocess, and, Get result:
    output_full_series, inputs = get_compensate_force(args, normed_data_X, normed_data_Y, model_part1, model_part2, part1_index, part2_index)

    #Evaluate here:
    #Note evaluation take place with normed data, and then denormed within.
    error_original, _, error_treated = evaluate.evaluate_error_rate(args, output_full_series, normed_data_Y, normer, raw_plan, showup=False)
    print("Error treated: {0:.2f}%, original:{1:.2f}%".format(error_treated, error_original))

    #Denormalize and Safety restrictions:
    #reference_full_series = raw_plan['need_to_compensate'].values
    compensate_full_series = normer.denormalize_Y(output_full_series)
    compensate_full_series = np.clip(compensate_full_series, -args.max_force, args.max_force)
    print(compensate_full_series)

    #To show:
    local_axis_num = args.axis_num - 1
    meassured = raw_plan['servo_feedback_torque_%s'%local_axis_num].values  #same as plan+target
    planned = raw_plan['axc_torque_ffw_gravity_%s'%local_axis_num].values
    compensated = planned + compensate_full_series

    #Visualize:
    if args.VISUALIZATION:
        plt.plot(meassured, label=r'real_target', alpha=0.5)
        plt.scatter(range(len(raw_plan)), planned, label=r'dynamic_model+gravity', color='gray', s=1, alpha=0.5)
        plt.scatter(part1_index, compensated[part1_index], label=r'speed', color='green', s=1)
        plt.scatter(part2_index, compensated[part2_index], label=r'after compensate', color='red', s=1)
        plt.legend()
        plt.title("Error treated: {0:.2f}%, original:{1:.2f}%".format(error_treated, error_original))
        print("One of the reason it may shows higher error rate than expected is because some uniform speed range be ommited during train/val evaluation")
        plt.show()

    np.savetxt("../output/NN_compensation.txt", compensate_full_series)
    print("Done")

    #response_surface = True
    response_surface = False
    if response_surface and args.VISUALIZATION:
        fig = plt.figure(figsize=(16,9))
        axe = fig.add_subplot(1,1,1,projection='3d')
        plot_utils.response_surface(axe, model_part1, inputs)
        plt.show()





