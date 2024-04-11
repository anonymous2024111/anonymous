import torch
import numpy as np
from scipy.sparse import *
import sys
sys.path.append('eva100/abalation/')
from mdataset import *

def norm(spmm):
    if spmm<10 :
        return '{:.2f}'.format(spmm)
    elif spmm < 100 :
        return '{:.1f}'.format(spmm)
    else:
        return '{:.0f}'.format(spmm)
    
# ablation-study
if __name__ == "__main__":
    
    dataset = ['reddit', 'ogb', 'AmazonProducts', 'IGB_medium', 'IGB_large']
    dimN = 64
    # #TF32
    # # CSV 文件路径
    file_name = './eva100/others/result/' + 'mma-motivation.txt'
    with open(file_name, 'w') as file:
        file.write('result : \n')
    for data in dataset:
        res = data
        
        # GAT - FP16
        # with-16x8
        inputInfo_16_gat = MGAT_dataset_m16()
        inputInfo_16_gat.m_block_16_8_r(data, dimN)

        
        
        # with-8x8
        inputInfo_8_gcn = MGCN_dataset_m16()
        inputInfo_8_gcn.m_block_8_8_r(data, dimN)

        
        mma_16 = round((inputInfo_16_gat.degrees.size(0)/128)*2,0)
        mma_8 = round((inputInfo_8_gcn.degrees.size(0))/64,0)
        
        res = res + ' & ' + ' & ' + str(mma_16)  
        res = res + ' & ' + ' & ' + ' & ' + str(mma_8)
    
        reduction  =  (mma_16 - mma_8) /mma_16
        
        res = res + ' & ' + ' & ' + str(round(reduction*100,2)) + '\%'
        res = res + ' \\\\ ' 

        with open(file_name, 'a') as file:
            file.write(res + '\n')
        print(data + ' is successed!')
print('success')



