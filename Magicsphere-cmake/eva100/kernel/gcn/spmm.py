import torch
from scipy.sparse import *
import sys
sys.path.append('eva100/kernel/gcn/')
from mtest import *
from mdataset import *
from advisor import test_advisor
from tcgnn import test_tcgnn
from gespmm import test_gespmm
from cusparse import test_cusparse
import csv

'''
GNNAdvisor
'''
def advisor_test(dataset, hidden, epoches, head ) : 
    
    csv_file = './eva100/kernel/gcn/result/advisor.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        # if data == 'AmazonProducts' : 
        #     res = []
        #     res.append(data)
        #     for dimN in hidden:
        #         res.append(str(-1))
        #         res.append(str(-1))
        #     # 写入 CSV 文件
        #     with open(csv_file, 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(res)
        #     continue
        res = []
        res.append(data)
        for dimN in hidden:         
            # res.append(str(dimN))
            # 计算
            spmm = test_advisor.test(data, epoches, dimN)
            res.append(str(spmm))

        # 写入 CSV 文件
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(res)

'''
TCGNN
'''
def tcgnn_test(dataset, hidden, epoches, head ) : 
    
    csv_file = './eva100/kernel/gcn/result/tcgnn.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        if data == 'AmazonProducts' : 
            res = []
            res.append(data)
            for dimN in hidden:
                res.append(str(1200))
            # 写入 CSV 文件
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
            continue
        if data == 'reddit' : 
            res = []
            res.append(data)
            for dimN in hidden:
                res.append(str(400))
            # 写入 CSV 文件
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
            continue
        # 计算量
        inputInfo_16 = MGCN_dataset_m16()
        inputInfo_16.m_block_16_8_mr(data, 64)
        compute_16 = inputInfo_16.num_edges
        
        res = []
        res.append(data)
        for dimN in hidden:
            # res.append(str(dimN))
            # 计算
            spmm = test_tcgnn.test(data, epoches, dimN)
            res.append(str(spmm))
        
            compute_n = compute_16 * dimN *2
            compute = round( compute_n/(spmm * 1e9), 4)
            res.append(str(compute))
            

        # 写入 CSV 文件
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(res)
        


'''
GESpMM
'''
def gespmm_test(dataset, hidden, epoches, head ) : 
    
    csv_file = './eva100/kernel/gcn/result/gespmm.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        if data=='IGB_medium':
            res = []
            res.append(data)
            for dimN in hidden:
                res.append(str(-1))
            # 写入 CSV 文件
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
            continue
        res = []
        res.append(data)
        for dimN in hidden:
            # res.append(str(dimN))
            # 计算
            spmm = test_gespmm.test(data, epoches, dimN)
            res.append(str(spmm))

        # 写入 CSV 文件
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(res)
            
'''
Cusparse
'''
def cusparse_test(dataset, hidden, epoches, head ) : 
    
    csv_file = './eva100/kernel/gcn/result/cusparse.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        res = []
        res.append(data)
        for dimN in hidden:
            # res.append(str(dimN))
            # 计算
            spmm = test_cusparse.test(data, epoches, dimN)
            res.append(str(spmm))

        # 写入 CSV 文件
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(res)


'''
M-tf32
'''
def m_tf32_test(dataset, hidden, epoches, head ) : 
    
    csv_file = './eva100/kernel/gcn/result/mtf32.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        res = []
        res.append(data)
        for dimN in hidden:
            # res.append(str(dimN))
            # 计算
            inputInfo_8_mr = MGCN_dataset_m32()
            inputInfo_8_mr.m_block_8_4_mr(data, dimN)
            # 8x1+MR v2
            spmm = test_tf32_v2(data, epoches, dimN, inputInfo_8_mr)
            res.append(str(spmm))
            
            # compute_fp16_8 = (inputInfo_8_mr.row_pointers[-1].item()) * 8 * dimN * 2
            compute_fp16_8 = inputInfo_8_mr.num_edges * dimN * 2
            compute = round( compute_fp16_8/(spmm * 1e9), 4)
            res.append(str(compute))

        # 写入 CSV 文件
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(res)     

'''
M-fp16
'''
def m_fp16_test(dataset, hidden, epoches, head ) : 
    
    csv_file = './eva100/kernel/gcn/result/mfp16.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        res = []
        res.append(data)
        for dimN in hidden:
            # res.append(str(dimN))
            # 计算
            inputInfo_8_mr = MGCN_dataset_m16()
            inputInfo_8_mr.m_block_8_8_mr(data, dimN)
            # 8x1+MR v2
            spmm = test_fp16_v2(data, epoches, dimN, inputInfo_8_mr)
            res.append(str(spmm))
            
            compute_fp16_8 = inputInfo_8_mr.num_edges * dimN * 2
            compute = round( compute_fp16_8/(spmm * 1e9), 4)
            res.append(str(compute))

        # 写入 CSV 文件
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(res)     
                      
if __name__ == "__main__":
    # 获取第一个可用的 GPU 设备
    gpu_device = torch.cuda.current_device()
    
    # 打印 GPU 设备的名称
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)
    
    dataset = ['AmazonProducts', 'reddit', 'ogb', 'IGB_medium']
    hidden = [64, 128, 256, 512]
    head= ['Dataset', '64', '128', '256', '512']
    head_compute = ['Dataset', '64', 'compute', '128', 'compute', '256', 'compute','512','compute']
    
    
    epoches = 10
    
    # # # Advisor
    advisor_test(dataset, hidden, epoches, head)
    print("Advisor success!")
    print()

        
    # #GESpMM
    gespmm_test(dataset, hidden, epoches, head)
    print("GESpMM success!")
    print()
        
    # #Mtf32
    m_tf32_test(dataset, hidden, epoches, head_compute)
    print("Mtf32 success!")
    print()
      
    # # #Mfp16
    m_fp16_test(dataset, hidden, epoches, head_compute)
    print("Mfp16 success!")
    print()
    
    # # #cusparse
    cusparse_test(dataset, hidden, epoches, head)
    print("cusparse success!")
    print()
       
    # #TC-GNN
    tcgnn_test(dataset, hidden, epoches, head_compute)
    print("TC-GNN success!")
    print()