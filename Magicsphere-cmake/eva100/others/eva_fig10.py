import torch
import seaborn as sns
from scipy.sparse import *
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('eva100/others/')

from mdataset import *


import pandas as pd

    
# ablation-study
if __name__ == "__main__":
    # # 获取第一个可用的 GPU 设备
    # gpu_device = torch.cuda.current_device()
    
    # # 打印 GPU 设备的名称
    # gpu = torch.cuda.get_device_name(gpu_device)
    # print(gpu)
    reload = False

    if reload:
        hidden = [64, 128, 256, 512]

        epoches = 10
        
        dataset = ['reddit', 'ogb', 'AmazonProducts', 'amazon', 'blog', 'artist']
        k = [3, 6, 9, 9, 9, 15]
        
        dataset_name = ['Reddit', 'OgbProducts', 'AmazonProducts', 'Amazon', 'Blog', 'Artist']
        dimN = 64
        res = dict()
        res['data'] = []
        res['num'] = []
        res['kernel'] = []
        res_k = dict()
        res_k['data'] = []
        res_k['num'] = []
        
        percen = []
        for kk, name in zip(k,dataset_name):
            res_k['data'].append(name)
            res_k['num'].append(kk)
        for data, name in zip(dataset, dataset_name):
            res['data'].append(name)
            # SpMM
            # with-8x4
            inputInfo_8_gcn = MGCN_dataset_m32()
            inputInfo_8_gcn.m_block_8_4_r(data, dimN)

            inputInfo_8_gcn_m = MGCN_dataset_m32()
            inputInfo_8_gcn_m.m_block_8_4_mr(data, dimN)
            
            mma_8_4_r = round((inputInfo_8_gcn.degrees.size(0)/32),0)
            mma_8_4_mr= round((inputInfo_8_gcn_m.degrees.size(0))/32,0)
            res['num'].append(mma_8_4_r - mma_8_4_mr)
            percen.append((mma_8_4_r - mma_8_4_mr)/mma_8_4_r)
            res['kernel'].append('SpMM')

            res['data'].append(name)
            # SDDMM
            # with-8x16
            inputInfo_8_gat = MGAT_dataset_m16()
            inputInfo_8_gat.m_block_8_16_r(data, dimN)

            inputInfo_8_gat_m = MGAT_dataset_m16()
            inputInfo_8_gat_m.m_block_8_16_mr(data, dimN)
            
            mma_8_16_r = round((inputInfo_8_gat.degrees.size(0)/128),0)
            mma_8_16_mr= round((inputInfo_8_gat_m.degrees.size(0))/128,0)
            res['num'].append(mma_8_16_r - mma_8_16_mr)
            res['kernel'].append('SDDMM')
                
        # 开始绘图：
        df = pd.DataFrame(res)
        df1 = pd.DataFrame(res_k)
        # print(df)
        df.to_csv('./eva100/others/result/rabbit_1.csv', index=False)
        df1.to_csv('./eva100/others/result/rabbit_2.csv', index=False)
        
        average = (sum(percen) / len(percen))*100
        print(average)
        print(percen)
    
    df = pd.read_csv('./eva100/others/result/rabbit_1.csv')
    df1 = pd.read_csv('./eva100/others/result/rabbit_2.csv')
    plt.figure(figsize=(5, 2))
    # 对数据按照 'dim' 列进行升序排序
    sns.set_style("darkgrid")
    # 设置背景色为灰白色

    g = sns.barplot(x='data', y='num', hue='kernel', data=df, palette='Blues_d', linewidth=1, legend=False)

    g.set_ylabel('')
    plt.ylim(1000, 10000000)  # 将 Y 轴下限设置为 40
    plt.yscale('log')  # y轴按照对数刻度显示
    plt.xticks(rotation=20)
    sns.set_style("white")

    plt.savefig('./eva100/others/result/mrabbit.png', dpi=800)
    # 清空图形
    plt.clf()
    print('success')