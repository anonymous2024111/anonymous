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
sys.path.append('Magicsphere/GCN/test')
from mgcn.mdataset_fp16 import *
import MagicsphereGCN_kernel
import time



def check(row_pointers1, column_index1, dd, rhs, n) :
    row_pointers1 = row_pointers1[:n+1]
    dd = dd.numpy()
    value = []
    for i in range(len(row_pointers1) - 1):
        for j in range(row_pointers1[i], row_pointers1[i+1]):
            value.append(dd[i]*dd[column_index1[j]])
    # n = row_pointers1.size(0)-1
    sparse_matrix = csr_matrix((value, column_index1.numpy(), row_pointers1.numpy()), shape=(n, n))
    result = sparse_matrix.dot(rhs.numpy())
    return result


def kernel(inputInfo, epoches, res, nOri, mOri):
     for i in range(epoches):
        X_prime, spmm_ms_avg  = MagicsphereGCN_kernel.forward_16(inputInfo.row_pointers, inputInfo.column_index, inputInfo.degrees, 
                               inputInfo.x, inputInfo.num_nodes, inputInfo.x.size(1), inputInfo.num_nodes_ori,100)
        print(round(spmm_ms_avg.item(),4))

        for i in range(mOri):
            if abs(X_prime[i][0] - res[i][0]) > 1 :
                print("No")
                exit(0)
            if abs(X_prime[i][nOri-1] - res[i][nOri-1]) > 1:
                print("No")
                exit(0)
        print("PASS")
def test(data, epoches, hidden):
    # 记录程序开始时间
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data)
    baseline = dict()
    spmm = dict()
    for dimN in hidden:
        baseline.clear()
        inputInfo.init_embedding(dimN)
        
        result = check(inputInfo.row_pointers1, inputInfo.column_index1, inputInfo.dd, inputInfo.x, inputInfo.num_nodes_ori)
        # inputInfo1 = inputInfo.to(device)

        kernel(inputInfo, epoches, result,  dimN, inputInfo.num_nodes_ori)
  
    return spmm


if __name__ == "__main__":
    dataset = 'cora'
    test(dataset, 1, [128])
   