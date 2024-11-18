import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('./Magicsphere/benchmark/GAT-benchmark/test')
from mgcn32_16.mdataset_fp32 import *
import MagicsphereGAT_kernel
import MagicsphereGAT2
import time





def kernel(inputInfo, epoches, nOri, mOri):
    X_prime, spmm_ms_avg =  MagicsphereGAT_kernel.forward_gen_tf32_16(inputInfo.x.size(1), inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, inputInfo.degrees, 
                            inputInfo.x, inputInfo.max, 100)
    print(round(spmm_ms_avg.item(),4))
    res = inputInfo.degrees * 128
    # X_prime=  MagicsphereGCN.forward_tf32_16(inputInfo.row_pointers, inputInfo.column_index, inputInfo.degrees, 
    #                        inputInfo.x, inputInfo.num_nodes, inputInfo.x.size(1), inputInfo.num_nodes_ori)
    for i in range(100):
        if abs(X_prime[i] - res[i]) > 1 :
            print("No")
            exit(0)
    print("PASS")
def test(data, epoches, hidden):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data)
    baseline = dict()
    spmm = dict()
    for dimN in hidden:
        baseline.clear()
        inputInfo.init_embedding(dimN)
           
        kernel(inputInfo, epoches,  dimN, inputInfo.num_nodes_ori)
  
    return spmm


if __name__ == "__main__":
    dataset = 'cora'
    test(dataset, 1, [128])
   