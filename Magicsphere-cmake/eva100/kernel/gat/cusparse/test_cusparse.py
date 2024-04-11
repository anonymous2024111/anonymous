import sys
sys.path.append('./eva100/kernel/gat')
from cusparse.mdataset import *
import cuSPARSE_kernel


def kernel(inputInfo, epoches):
    X_prime, spmm_ms_avg = cuSPARSE_kernel.cuSPARSE_SDDMM_CSR(inputInfo.row_pointers, inputInfo.column_index, inputInfo.x, inputInfo.num_nodes, inputInfo.x.size(1),inputInfo.num_edges, epoches, 10)
    return round(spmm_ms_avg.item(),4)

  
def test(data, epoches, dimN):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = GCN_dataset(data)
    baseline = dict()
    spmm = dict()
    
    baseline.clear()
    inputInfo.init_embedding(dimN)
    inputInfo.to(device)
    execution_time = kernel(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' cusparse-' + str(execution_time))

    return execution_time



   