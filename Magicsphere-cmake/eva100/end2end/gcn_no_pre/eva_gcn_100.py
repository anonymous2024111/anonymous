import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./eva100/end2end/gcn_no_pre')
from advisor import test_advisor
from mydgl import test_dgl
from mypyg import test_pyg
from mgcn import test_mgcn
from tcgnn import test_tcgnn
from mgcn32 import test_mgcn32
#DGL
def dglGCN(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_dgl.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-dgl-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)   
    
#Advisor
def advisor(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_advisor.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-advisor-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)   
    
#Tcgnn
def tcgnn(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_tcgnn.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-tcgnn-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    
#MGCN
def mGCN16(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_mgcn.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgcn-fp16-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  

#MGCN-tf32
def mGCN32(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm= test_mgcn32.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgcn-tf32-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    
#MGPYG
def pygGCN(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_pyg.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-pyg-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  



if __name__ == "__main__":


    dataset_I = ['Coauthor_Physics',  'FacebookPagePage', 'email-Enron', 'loc-Brightkite', 'soc-Epinions1', 
                 'HR_NO', 'HU_NO', 'GitHub', 'artist', 'blog']

    dataset_II = ['ell',  'com-DBLP', 'Reddit2', 'amazon', 'amazon0505', 
                  'dd', 'yelp', 'comamazon', 'roadNet-CA', 'roadNet-PA', 
                  'roadNet-TX', 'yeast', 'DGraphFin', ]
    dataset_III = ['reddit', 'ogb', 'AmazonProducts']
    
    #3-64: 
    dataset_3_64 = dataset_II + dataset_III
    #6-64 :
    dataset_6_64 = dataset_II
    #layer == 6 and dim>=128:
    dataset_6 = dataset_I + dataset_II
    #others: 
    dataset = dataset_I + dataset_II + dataset_III


    layer = [3, 6]
    hidden = [64, 128, 256]
    # dataset = ['cora', 'cite']
    # layer = [3]
    # hidden = [512]

    epoches = 300
    featuredim = 512
    classes = 10
    
    start = time.time()
    #DGL
    filename = './eva100/end2end/gcn_no_pre/result/dgl-v3.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            if hidden_num == 64 and layer_num==3 :
                for data in dataset_3_64:
                    dglGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif hidden_num == 64 and layer_num==6 :
                for data in dataset_6_64:
                    dglGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif layer_num == 6:
                for data in dataset_6:
                    dglGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            else:
                for data in dataset:
                    dglGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)

    end = time.time()
    print('DGL-' + 'success using time (s) : ' + str(end-start))
    
    start = time.time()       
    #MGCN-fp16
    filename = './eva100/end2end/gcn_no_pre/result/mgcn16-v2.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            if hidden_num == 64 and layer_num==3 :
                for data in dataset_3_64:
                    mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif hidden_num == 64 and layer_num==6 :
                for data in dataset_6_64:
                    mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif layer_num == 6:
                for data in dataset_6:
                    mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            else:
                for data in dataset:
                    mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)           
    end = time.time()
    print('MGCN-fp16-' + 'success using time (s) : ' + str(end-start))
    
    start = time.time()
    #MGCN-tf32   
    filename = './eva100/end2end/gcn_no_pre/result/mgcn32-v2.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            if hidden_num == 64 and layer_num==3 :
                for data in dataset_3_64:
                    mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif hidden_num == 64 and layer_num==6 :
                for data in dataset_6_64:
                    mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif layer_num == 6:
                for data in dataset_6:
                    mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            else:
                for data in dataset:
                    mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    end = time.time()
    print('MGCN-tf32-' + 'success using time (s) : ' + str(end-start))
    
    start = time.time()
    #Advisor
    filename = './eva100/end2end/gcn_no_pre/result/advisor-v3.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            if hidden_num == 64 and layer_num==3 :
                for data in dataset_3_64:
                    advisor(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif hidden_num == 64 and layer_num==6 :
                for data in dataset_6_64:
                    advisor(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif layer_num == 6:
                for data in dataset_6:
                    advisor(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            else:
                for data in dataset:
                    advisor(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    end = time.time()
    print('Advisor-' + 'success using time (s) : ' + str(end-start))

    #TCGNN
    filename = './eva100/end2end/gcn_no_pre/result/tcgnn-v3.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            if hidden_num == 64 and layer_num==3 :
                for data in dataset_3_64:
                    tcgnn(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif hidden_num == 64 and layer_num==6 :
                for data in dataset_6_64:
                    tcgnn(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif layer_num == 6:
                for data in dataset_6:
                    tcgnn(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            else:
                for data in dataset:
                    tcgnn(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('TCGNN-' + 'success')
    
    #PYG
    filename = './eva100/end2end/gcn_no_pre/result/pyg-v3.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            if hidden_num == 64 and layer_num==3 :
                for data in dataset_3_64:
                    pygGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif hidden_num == 64 and layer_num==6 :
                for data in dataset_6_64:
                    pygGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            elif layer_num == 6:
                for data in dataset_6:
                    pygGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
            else:
                for data in dataset:
                    pygGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)

    # print('Pyg-' + 'success')

    # print('MGCN_all success')