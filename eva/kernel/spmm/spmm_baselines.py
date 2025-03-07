import torch
from scipy.sparse import *
import sys
# from advisor import test_advisor
from tcgnn import test_tcgnn
from gespmm import test_gespmm
import subprocess

import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

            
'''
GNNAdvisor
'''
def advisor_test(data, dimN, epoches,data_path) : 
    spmm = test_advisor.test(data, epoches, dimN, data_path)
    return spmm
                
            
'''
TCGNN
'''
def tcgnn_test(data, dimN, epoches,data_path) : 
    spmm = test_tcgnn.test(data, epoches, dimN, data_path)
    return spmm

           
''' 
GE-SpMM
'''
def gespmm_test(data, dimN, epoches,data_path) : 
    spmm = test_gespmm.test(data, epoches, dimN, data_path)
    return spmm


def safe_advisor_test(data, dimN, epoches, data_path):
    try:
        result = subprocess.check_output(
            ['python3', 'advisor/test_advisor.py', data, str(dimN), str(epoches), data_path],
            stderr=subprocess.STDOUT
        )
        return float(result.decode().strip())  # 假设返回值是一个数字
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed: {e.output.decode()}")
        return 100000  # 默认值
    
    
if __name__ == "__main__":

    gpu_device = torch.cuda.current_device()
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)


    dimN = int(sys.argv[1])
    print('dimN: ' + str(dimN))
    epoches = 10
    current_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    #result path
    file_name = project_dir + '/result/Baseline/spmm/base_spmm_f32_n' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'advisor', 'tcgnn', 'gespmm']
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)
    
    start_time = time.time()
    # Traverse each dataset
    df = pd.read_csv(project_dir + '/dataset/data_filter.csv')
    df = pd.read_csv(project_dir + '/result/ref/baseline_h100_spmm_256.csv')
    
    for index, row in df.iterrows():
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])

        #Dataset path
        data_path =  project_dir + '/dataset/' + row.iloc[0] + '.npz'
        
        # advisor
        # if row.iloc[0] not in []:
        #     spmm_advisor = advisor_test(row.iloc[0], dimN, epoches, data_path)
        #     res_temp.append(spmm_advisor)
        # else:
        #     spmm_advisor = advisor_test(row.iloc[0], dimN, epoches, data_path)
        #     res_temp.append(spmm_advisor)
        spmm_advisor = safe_advisor_test(row.iloc[0], dimN, epoches, data_path)
        print(str(dimN) + '-' + row.iloc[0] + ' advisor-' + str(spmm_advisor))
        res_temp.append(spmm_advisor)
        
        # tcgnn
        if row.iloc[2] < 1000000:
            spmm_tcgnn = tcgnn_test(row.iloc[0], dimN, epoches, data_path)
            res_temp.append(spmm_tcgnn)
        else:
            res_temp.append(10000000)
            
        # gespmm
        spmm_gespmm = gespmm_test(row.iloc[0], dimN, epoches, data_path)
        res_temp.append(spmm_gespmm)
            
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)
        print(row.iloc[0] + ' is success')
        print()
    print('All is success')
    
    end_time = time.time()
    execution_time = end_time - start_time

    # Record execution time.
    with open("execution_time_base.txt", "a") as file:
        file.write("Baseline-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")