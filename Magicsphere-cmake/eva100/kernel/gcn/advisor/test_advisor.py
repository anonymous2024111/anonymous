import os
import sys
# sys.path.append('./eva100/kernel/gcn')
from advisor.mdataset import *
from advisor.param import *
import GNNAdvisor_v2_kernel

def kernel(inputInfo, epoches, dataset):
    # for i in range(epoches):
    X_prime, spmm_ms_avg = GNNAdvisor_v2_kernel.forward(dataset.x, inputInfo.row_pointers, inputInfo.column_index, inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node, inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock, epoches, 10)
    return round(spmm_ms_avg.item(),4)

def test(data, epoches, dimN):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = GCN_dataset(data, dimN)
    dataset.to(device)
    partSize = 32
    dimWorker = 32
    warpPerBlock = 4
    sharedMem = 100
    column_index = dataset.column_index
    row_pointers = dataset.row_pointers
    degrees = dataset.degrees
    inputInfo = inputProperty(row_pointers, column_index, degrees, 
                            partSize, dimWorker, warpPerBlock, sharedMem,
                            hiddenDim=dimN,dataset_obj=dataset)
    inputInfo.decider()
    inputInfo = inputInfo.set_input()
    inputInfo = inputInfo.set_hidden()
    partPtr, part2Node = GNNAdvisor_v2_kernel.build_part(inputInfo.partSize, inputInfo.row_pointers)
    inputInfo.row_pointers  = inputInfo.row_pointers.to(device)
    inputInfo.column_index  = inputInfo.column_index.to(device)
    inputInfo.partPtr = partPtr.int().to(device)
    inputInfo.part2Node  = part2Node.int().to(device)

    execution_time = kernel(inputInfo, epoches, dataset)
    print(str(dimN) + '-' + data + ' advisor-' + str(execution_time))
    return execution_time
