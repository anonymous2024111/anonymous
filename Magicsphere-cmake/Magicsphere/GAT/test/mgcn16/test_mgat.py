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
sys.path.append('Magicsphere/GAT/test')
from mgcn16.mdataset_fp32 import *
import MagicsphereGAT_cmake
import MagicsphereGCN_cmake
import time





def kernel(inputInfo, epoches, nOri, mOri):
        att =  MagicsphereGAT_cmake.fp16_sddmm_csr(inputInfo.x.size(1), inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, inputInfo.degrees, 
                               inputInfo.x, inputInfo.x, inputInfo.max, inputInfo.num_edges)

        output=MagicsphereGCN_cmake.forward_v2_csr(inputInfo.row_pointers, inputInfo.column_index, inputInfo.templete, att,  inputInfo.x, inputInfo.num_nodes, inputInfo.x.size(1), inputInfo.num_nodes_ori)
        print()

        # test = inputInfo.x[inputInfo.column_index1[0:31],:]
        print()


def test(data, epoches, hidden):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data)
    baseline = dict()
    spmm = dict()
    for dimN in hidden:
        baseline.clear()
        inputInfo.init_embedding(dimN)
        inputInfo = inputInfo.to(device)
        kernel(inputInfo, epoches,  dimN, inputInfo.num_nodes_ori)
  
    return spmm


if __name__ == "__main__":
    dataset = 'cora'
    test(dataset, 1, [128])
   