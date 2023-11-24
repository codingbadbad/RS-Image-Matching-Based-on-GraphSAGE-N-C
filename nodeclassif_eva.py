import csv
import time

import cv2
import numpy as np
import pandas as pd
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data, Batch
import csv
import time
import re
from torch_geometric.data import DataListLoader
import random
import os
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import  torch
from tqdm import tqdm

import dataloader_for_node
import dataloader_for_node as dl
from load import DiabetesDataset ,load_datasets , GraphSAGENet ,load_single_dataset
from torch.nn import Linear


import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
def readgt(file_path, file_name, th=62):
    if file_name.endswith('.csv'):

        data = pd.read_csv(file_path+file_name, header=None)
        extracted_data = []
        x = []
        y = []

        CorrectIndex = []

        for i in range(0, len(data)):

            values = data.iloc[i, :].values
            vector1 = np.array([values[0], values[1]])
            vector2 = np.array([values[2], values[3]])



            x.append([values[4], values[5]])
            y.append([values[6], values[7]])


        x_data = np.array(x)
        y_data = np.array(y)


        return  x_data , y_data









bias = 0.5
def extract_png_names(filename ,endwith = '.jpg'):
    pattern = r"(\w+\.jpg)"

    if endwith  == '.png' :pattern = r"(\w+\.png)"
    # pattern = r"(\w+\.jpg)"
    png_names = re.findall(pattern, filename)
    png_names = [name.replace('from', '') for name in png_names]
    return png_names

def readname (th , file_name , endswith):
    if th == 1:
        match = re.search(r"from(\d+)_", file_name)
        k = match.group(1)
        print(k)
        image_path1 = k + 'l' + '.png'
        image_path2 = k + 'r' + '.png'
    else:
        [image_path1,image_path2] = extract_png_names(file_name, endwith= endswith)
         # = extract_png_names(file_name)[1]
    return image_path1,image_path2

def draw (path,filename , pred  ,inter = 62):
    image_path1, image_path2 = readname(1,filename , endswith = '.png')  ## for rs
    # image_path1, image_path2 = readname(0,filename , endswith = '.jpg')    ## for pool
    # image_path1, image_path2 = readname(0,filename , endswith = '.png')  ## fisheye
    image1 = cv2.imread(path+ image_path1)
    image2 = cv2.imread(path+ image_path2)


# ### 超分辨率
#     image1 = upper.super_resolve_image(path+ image_path1)
#     image2 = upper.super_resolve_image(path+ image_path2)

    # cv2.imshow("",image1)
    # cv2.waitKey(0)
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    # print(height1,width1,height2,width2)

    hmax = max(height1, height2)
    wmax = max(width1, width2)

    merged_width = width1 + width2
    merged_height = max(height1, height2)
    shift = 10
    # 创建一个新的空白图像作为拼接结果
    merged_image = np.zeros((merged_height, merged_width +shift, 3), dtype=np.uint8) + 255

    # 将第一张图像复制到拼接图像的左侧
    merged_image[:height1, :width1] = image1

    # 将第二张图像复制到拼接图像的右侧
    merged_image[:height2, width1+shift:] = image2

    whiteimage = np.zeros((hmax, wmax, 3), dtype=np.uint8) +255

    xy = np.loadtxt(path+ filename, delimiter=',', dtype=np.float32)
    labels = xy[::inter, 0]  # 假设每隔31行取一个label
    pred = pred.cpu().numpy()  # 将预测值从torch张量转换为numpy数组

    labels = labels.astype(int)
    pred = pred.astype(int)


    results = []

    # 对每个样本进行判断
    for p, l in zip(pred, labels):
        if p == 1 and l == 1:
            results.append('TP')
        elif p == 1 and l == 0:
            results.append('FP')
        elif p == 0 and l == 1:
            results.append('FN')
        else:  # p == 0 and l == 0
            results.append('TN')

    # 输出结果
    # print(len(labels) , len(pred))
    # for i, result in enumerate(results):
    #     print(f"Sample {i + 1}: {result}")
    x = xy[::inter, [3,4]]  # 假设每隔31行取一个label
    y = xy[::inter, [5,6]]  # 假设每隔31行取一个label
    width1 += shift
    for i in range(0, len(results),2):

        if results[i] == 'TN':
            pt1 = (int(x[i][0] * wmax),          int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_= (int(y[i][0] * wmax),          int(y[i][1] * hmax))
            # cv2.line(merged_image, pt1, pt2,  (0, 0, 0), 1)
            cv2.arrowedLine(whiteimage,   pt1, pt2_, (0, 0, 0), 1,tipLength = 0.05 ,line_type=16)

    # for i in range(0, len(results),2):
    for i in range(0, len(results)):
        if results[i] == 'TP':
            pt1 = (int(x[i][0] * wmax),          int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_= (int(y[i][0] * wmax),          int(y[i][1] * hmax))
            cv2.line(merged_image, pt1, pt2,  (255, 0, 0), 1 ,lineType=16)
            cv2.arrowedLine(whiteimage,   pt1, pt2_, (255, 0, 0), 1 ,tipLength = 0.05 ,line_type=16)
    for i in range( len(results)):
        if results[i] == 'FP':
            pt1 = (int(x[i][0] * wmax),          int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_= (int(y[i][0] * wmax),          int(y[i][1] * hmax))
            cv2.line(merged_image, pt1, pt2,  (0, 0, 255), 1,lineType=16)
            cv2.arrowedLine(whiteimage,   pt1, pt2_, (0, 0, 255), 1,tipLength = 0.05 ,line_type=16)
        if results[i] == 'FN':
            pt1 = (int(x[i][0] * wmax),          int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_= (int(y[i][0] * wmax),          int(y[i][1] * hmax))
            cv2.line(merged_image, pt1, pt2,  (0, 255, 0), 1,lineType=16)
            cv2.arrowedLine(whiteimage,   pt1, pt2_, (0, 255, 0), 1,tipLength = 0.05 ,line_type=16)
    #
    cv2.imshow('whiteimage', whiteimage)
    cv2.imshow('merged_image', merged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./view/'+image_path1+ image_path2+'whiteimage.png',whiteimage)
    cv2.imwrite('./view/'+image_path1+ image_path2+'merged_image.png',merged_image)


def eva_rmse(filename,gt_filepath,pred):
    #
    match = re.search(r"from(\d+)_", filename)  # for rs
    k = match.group(1)
    gtname = "gt"+str(k) +".csv"
    #
    # image_name1, image_name2 = readname(0, filename, endswith='.jpg')  ###for pool
    # gtname = "gt"+image_name1+image_name2+".csv"
    # print(image_name1,image_name2)



    xp ,yp = readgt(gt_filepath,gtname )
    # print(xp,yp)
    pred = pred.cpu().numpy()  # 将预测值从torch张量转换为numpy数组
    pred = pred.astype(int)

    print(len(xp ) , len(pred))
    pred_flat = pred.flatten()

    errors = xp[pred_flat == 1, :] - yp[pred_flat == 1, :]
    # print(errors)

    euclidean_distances = np.linalg.norm(errors, axis=1)
    euclidean_sum = np.sum(errors ** 2, axis=1)

    rmse = np.sqrt(np.mean(euclidean_sum))

    mae = np.max(euclidean_distances)
    mee = np.median(euclidean_distances)

    print(rmse,mae,mee)

    return rmse, mee, mae

def evaluatePR(outputs, labels):
    _, preds = outputs.max(1)

    # outputs = outputs.view(-1)
    # preds = (outputs > bias).float()

    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)

    tp = ((preds == 1) & (labels == 1)).sum().item()  # True Positive
    fn = ((preds == 0) & (labels == 1)).sum().item()  # False Negative
    fp = ((preds == 1) & (labels == 0)).sum().item()  # False Positive
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0


    return precision, recall , preds

def evaluate(model, test_loader):
    model.eval()

    with torch.no_grad():
        test_accuracy = 0
        test_recall = 0
        start_time = time.time()
        for test_data in test_loader:
            test_data = test_data.to(device)
            test_out = model(test_data.x, test_data.edge_index ,  None)

            accuracy, recall , pred= evaluatePR(test_out, test_data.y)
            # print(accuracy,recall)
            test_accuracy += accuracy
            test_recall += recall

        runtime = time.time() - start_time

        test_accuracy /= len(test_loader)
        test_recall /= len(test_loader)
        print([test_accuracy, test_recall])
        return test_accuracy, test_recall, runtime,pred


def evaluate_and_write_to_csv(model,train_loader , csv_writer ,file_name, timecost = 0):

    precision, recall ,runtime, pred = evaluate(model, train_loader)
    runtime = 0
    draw(path , file_name , pred)
    # csv_writer.writerow([file_name,precision, recall, runtime])

    # rmse, mee, mae = eva_rmse(file_name,gtpath,pred)
    # csv_writer.writerow([file_name,precision, recall, runtime + timecost,rmse, mee, mae])
    # csv_writer.writerow([file_name,precision, recall, runtime + timecost])






path = r'./test/'




files = [f for f in os.listdir(path) if f.endswith('.csv')]


device = torch.device('cuda')
# device = torch.device('cpu')

model = GraphSAGENet(in_channels=14, hidden_channels=54, out_channels=2).to(device)  # 输出层设为2
model.load_state_dict(torch.load('./saved_models/model_epoch_2465 +score 1540.4089634273905.pt'))
model.eval()

## double
with open('./graphsage_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename','Precision', 'Recall', 'Runtime','rmse',' mee', 'mae'])
    for file_name in files:

        data = load_single_dataset(path+file_name)

        train_loader = DataLoader([data], batch_size=1, shuffle=False)
        evaluate_and_write_to_csv(model, train_loader, writer ,file_name)

