import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv


type = 'v'
# 从csv中读取数据
res = dict()
data  = dict()
dataset = ['reddit', 'ogb', 'AmazonProducts', 'IGB_medium']
hidden = [64, 128, 256, 512]
# MGCNtf32
with open('./eva100/kernel/gcn/result/mtf32' + type + '.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        dim = dict()
        temp = dict()
        for i in range(4):
            temp['time'] = float(row[1+i*2])
            temp['compute'] = float(row[1+i*2+1])
            dim[hidden[i]] = temp.copy()
        data[dataset[cur]] = dim.copy()
    res['Magicsphere-tf32'] =data.copy()
    data.clear()
    
# MGCNfp16
with open('./eva100/kernel/gcn/result/mfp16' + type + '.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        dim = dict()
        temp = dict()
        for i in range(4):
            temp['time'] = float(row[1+i*2])
            temp['compute'] = float(row[1+i*2+1])
            dim[hidden[i]] = temp.copy()
        data[dataset[cur]] = dim.copy()
    res['Magicsphere-fp16'] =data.copy()
    data.clear()

# TCGNN
with open('./eva100/kernel/gcn/result/tcgnn' + type + '.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        dim = dict()
        temp = dict()
        for i in range(4):
            temp['time'] = float(row[1+i*2])
            temp['compute'] =float(row[1+i*2+1])
            dim[hidden[i]] = temp.copy()
        data[dataset[cur]] = dim.copy()
    res['TC-GNN'] =data.copy()
    data.clear()

# 下面无compute:
# GESpmm 
# with open('./eva100/kernel/gcn/result/gespmm' + type + '.csv-', 'r') as file:
#     reader = csv.reader(file, delimiter=',')
#     next(reader)  # 跳过第一行
#     for cur, row in enumerate(reader, start=0):
#         dim = dict()
#         temp = dict()
#         for i in range(4):
#             temp['time'] = float(row[1+i])
#             dim[hidden[i]] = temp.copy()
#         data[dataset[cur]] = dim.copy()
#     res['GE-SpMM'] =data.copy()
#     data.clear()   

# Advisor
with open('./eva100/kernel/gcn/result/advisor' + type + '.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        dim = dict()
        temp = dict()
        for i in range(4):
            temp['time'] = float(row[1+i])
            dim[hidden[i]] = temp.copy()
        data[dataset[cur]] = dim.copy()
    res['GNNAdvisor'] =data.copy()
    data.clear()   

# Cusparse 
# with open('./eva100/kernel/gcn/result/cusparse' + type + '.csv', 'r') as file:
#     reader = csv.reader(file, delimiter=',')
#     next(reader)  # 跳过第一行
#     for cur, row in enumerate(reader, start=0):
#         dim = dict()
#         temp = dict()
#         for i in range(4):
#             temp['time'] = float(row[1+i])
#             dim[hidden[i]] = temp.copy()
#         data[dataset[cur]] = dim.copy()
#     res['cusparse'] =data.copy()
#     data.clear()   


# 计算TC-GNN
baseline = ['TC-GNN']
print(baseline[0])
final = dict()
final['data']=[]
final['baseline']=[]
final['speed']=[]
for data in dataset:
    speedup = dict()
    speedup['dim'] = []
    speedup['speed'] = []
    speedup['baseline'] = []

    for dimN in hidden:
        for base in baseline:
            speedup['dim'].append(str(dimN))
            speed = round(  res[base][data][dimN]['time'] / res['Magicsphere-tf32'][data][dimN]['time'],2 )
            speedup['speed'].append(speed)
            speedup['baseline'].append('Magicsphere-tf32')
            
    for dimN in hidden:
        for base in baseline:
            speedup['dim'].append(str(dimN))
            speed = round(  res[base][data][dimN]['time'] / res['Magicsphere-fp16'][data][dimN]['time'],2 )
            speedup['speed'].append(speed)
            speedup['baseline'].append('Magicsphere-fp16')
            # speedup['color'].append(color)
    
            
    df = pd.DataFrame(speedup)
    final['data'].append(data)
    final['speed'].append(df.loc[df['baseline'] == 'Magicsphere-tf32', 'speed'].mean())
    final['baseline'].append('Magicsphere-tf32')
    
    final['data'].append(data)
    final['speed'].append(df.loc[df['baseline'] == 'Magicsphere-fp16', 'speed'].mean())
    final['baseline'].append('Magicsphere-fp16')
    # print()

df = pd.DataFrame(final)
# print(df)
print("tf32:" + str(round((df.loc[df['baseline'] == 'Magicsphere-tf32', 'speed'].mean()),2)))
print("tf32-max:" + str(round((df.loc[df['baseline'] == 'Magicsphere-tf32', 'speed'].max()),2)))
print("fp16:" + str(round((df.loc[df['baseline'] == 'Magicsphere-fp16', 'speed'].mean()),2)))
print("fp16-max:" + str(round((df.loc[df['baseline'] == 'Magicsphere-fp16', 'speed'].max()),2)))
print("avg:" + str(round((df['speed'].mean()),2)))


# 计算GNNAdvisor
baseline = ['GNNAdvisor']
print(baseline[0])
final = dict()
final['data']=[]
final['baseline']=[]
final['speed']=[]
for data in dataset:
    speedup = dict()
    speedup['dim'] = []
    speedup['speed'] = []
    speedup['baseline'] = []

    for dimN in hidden:
        for base in baseline:
            speedup['dim'].append(str(dimN))
            speed = round(  res[base][data][dimN]['time'] / res['Magicsphere-tf32'][data][dimN]['time'],2 )
            speedup['speed'].append(speed)
            speedup['baseline'].append('Magicsphere-tf32')
            
    for dimN in hidden:
        for base in baseline:
            speedup['dim'].append(str(dimN))
            speed = round(  res[base][data][dimN]['time'] / res['Magicsphere-fp16'][data][dimN]['time'],2 )
            speedup['speed'].append(speed)
            speedup['baseline'].append('Magicsphere-fp16')
            # speedup['color'].append(color)
    
            
    # 开始绘图：
    df = pd.DataFrame(speedup)
    final['data'].append(data)
    final['speed'].append(df.loc[df['baseline'] == 'Magicsphere-tf32', 'speed'].mean())
    final['baseline'].append('Magicsphere-tf32')
    
    final['data'].append(data)
    final['speed'].append(df.loc[df['baseline'] == 'Magicsphere-fp16', 'speed'].mean())
    final['baseline'].append('Magicsphere-fp16')
    # print()

df = pd.DataFrame(final)
# print(df)
print("tf32:" + str(round((df.loc[df['baseline'] == 'Magicsphere-tf32', 'speed'].mean()),2)))
print("tf32-max:" + str(round((df.loc[df['baseline'] == 'Magicsphere-tf32', 'speed'].max()),2)))
print("fp16:" + str(round((df.loc[df['baseline'] == 'Magicsphere-fp16', 'speed'].mean()),2)))
print("fp16-max:" + str(round((df.loc[df['baseline'] == 'Magicsphere-fp16', 'speed'].max()),2)))
print("avg:" + str(round((df['speed'].mean()),2)))