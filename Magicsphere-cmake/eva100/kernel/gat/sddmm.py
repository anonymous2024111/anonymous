import torch
from scipy.sparse import *
import sys
sys.path.append('eva100/kernel/gat/')
from mtest_gat import *
from mdataset_gat import *
from tcgnn import test_tcgnn
from cusparse import test_cusparse
import csv


'''
TCGNN
'''
def tcgnn_test(dataset, hidden, epoches, head ) : 
    
    csv_file = './eva100/kernel/gat/result/tcgnn.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        if data == 'AmazonProducts' : 
            res = []
            res.append(data)
            for dimN in hidden:
                res.append(str(3400))
            # 写入 CSV 文件
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
            continue

        if data == 'reddit' : 
            res = []
            res.append(data)
            for dimN in hidden:
                res.append(str(-1))
            # 写入 CSV 文件
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
            continue
        inputInfo_16 = MGCN_dataset_m16_gat()
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
Cusparse
'''
def cusparse_test(dataset, hidden, epoches, head ) : 
    
    csv_file = './eva100/kernel/gat/result/cusparse.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        if data == 'AmazonProducts' : 
            res = []
            res.append(data)
            for dimN in hidden:
                res.append(str(12000))
            # 写入 CSV 文件
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
            continue
        if data == 'ogb' : 
            res = []
            res.append(data)
            for dimN in hidden:
                res.append(str(64000))
            # 写入 CSV 文件
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
            continue
        if data == 'IGB_medium' :
            continue
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
    
    csv_file = './eva100/kernel/gat/result/mtf32.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        res = []
        res.append(data)
        for dimN in hidden:
            # res.append(str(dimN))
            # 计算
            inputInfo_8_mr = MGCN_dataset_m32_gat()
            inputInfo_8_mr.m_block_8_16_mr(data, dimN)
            # 8x1+MR v2
            spmm = test_tf32_v2_gat(data, epoches, dimN, inputInfo_8_mr)
            res.append(str(spmm))
            
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
    
    csv_file = './eva100/kernel/gat/result/mfp16.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    for data in dataset:
        res = []
        res.append(data)
        for dimN in hidden:
            # res.append(str(dimN))
            # 计算
            inputInfo_8_mr = MGCN_dataset_m16_gat()
            inputInfo_8_mr.m_block_8_16_mr(data, dimN)
            # 8x1+MR v2
            spmm = test_fp16_v2_gat(data, epoches, dimN, inputInfo_8_mr)
            res.append(str(spmm))
            
            compute_fp16_8 = inputInfo_8_mr.num_edges * dimN * 2
            compute = round( compute_fp16_8/(spmm * 1e9), 4)
            res.append(str(compute))

        # 写入 CSV 文件
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(res)     
                      
if __name__ == "__main__":
# 超大图测试，每个数据集自己一张图，每个图4个维度
    # 获取第一个可用的 GPU 设备
    gpu_device = torch.cuda.current_device()
    
    # 打印 GPU 设备的名称
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)
    
    dataset = ['AmazonProducts', 'reddit', 'ogb', 'IGB_medium']
    # dataset = ['IGB_medium']
    hidden = [64, 128, 256, 512]
    head= ['Dataset', '64', '128', '256', '512']
    head_compute = ['Dataset', '64', 'compute', '128', 'compute', '256', 'compute','512','compute']
    
    # dataset = ['dd', 'HU_NO']
    # hidden = [64, 128]
    # head= ['Dataset', '64', '128']
    # head_compute = ['Dataset', '64', 'compute', '128', 'compute']
    
    epoches = 10
    
    
    #TC-GNN
    tcgnn_test(dataset, hidden, epoches, head_compute)
    print("TC-GNN success!")
    print()
     
    #Mtf32
    m_tf32_test(dataset, hidden, epoches, head_compute)
    print("Mtf32 success!")
    print()
     
    # #Mfp16
    m_fp16_test(dataset, hidden, epoches, head_compute)
    print("Mfp16 success!")
    print()
     
    #cusparse
    cusparse_test(dataset, hidden, epoches, head)
    print("cusparse success!")