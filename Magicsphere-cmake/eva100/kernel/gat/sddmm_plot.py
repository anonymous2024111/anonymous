import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv



# 从csv中读取数据
res = dict()
data  = dict()
dataset = ['AmazonProducts', 'reddit',  'ogb', 'IGB_medium']
dataset_cu = ['AmazonProducts', 'reddit',  'ogb']
hidden = [64, 128, 256, 512]
# MGATtf32
with open('./eva100/kernel/gat/result/mtf32' + type + '.csv', 'r') as file:
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
    
# MGATfp16
with open('./eva100/kernel/gat/result/mfp16' + type + '.csv', 'r') as file:
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
with open('./eva100/kernel/gat/result/tcgnn' + type + '.csv', 'r') as file:
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


# Cusparse 
with open('./eva100/kernel/gat/result/cusparse' + type + '.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        dim = dict()
        temp = dict()
        for i in range(4):
            temp['time'] = float(row[1+i])
            dim[hidden[i]] = temp.copy()
        data[dataset_cu[cur]] = dim.copy()
    res['cusparse'] =data.copy()
    data.clear()   

#依此绘制每个图
baseline = ['TC-GNN','Magicsphere-tf32', 'Magicsphere-fp16']

mycolor = {'TC-GNN':'limegreen', 'GE-SpMM':'orchid', 'GNNAdvisor':'coral', 
           'Magicsphere-tf32':'cornflowerblue', 'Magicsphere-fp16':'royalblue'}
baseline2 = ['TC-GNN', 'Magicsphere-tf32', 'Magicsphere-fp16']

for data in dataset:
    if data == 'IGB_medium':
        continue
    speedup = dict()
    speedup['dim'] = []
    speedup['speed'] = []
    speedup['baseline'] = []
    # speedup['color'] = []

    compute = dict()
    compute['dim'] = []
    compute['TC-GNN'] = []
    compute['Magicsphere-tf32'] = []
    compute['Magicsphere-fp16'] = []
    for dimN in hidden:
        for base, color in zip(baseline, mycolor):
            speedup['dim'].append(str(dimN))
            if res[base][data][dimN]['time'] == -1:
                speedup['speed'].append(0)
            else :
                speed = round( res['cusparse'][data][dimN]['time'] / res[base][data][dimN]['time'],2 )
                speedup['speed'].append(speed)
            speedup['baseline'].append(base)
            # speedup['color'].append(color)
    

    for dimN in hidden:
        compute['dim'].append(str(dimN))
        for base in baseline2:
            if res[base][data][dimN]['compute'] == -1 :
                compute[base].append(0)
            else:
                compute[base].append(round( res[base][data][dimN]['compute'],2 ))

            
            
    # 开始绘图：
    df = pd.DataFrame(speedup)
    df1 = pd.DataFrame(compute)
    # print(df)

    
    # sns.set_style("whitegrid")

    # 对数据按照 'dim' 列进行升序排序
    sns.set_style("darkgrid")
    # 设置背景色为灰白色

    g = sns.barplot(x='dim', y='speed', hue='baseline', data=df, palette=mycolor, linewidth=1, legend=False)
    plt.axhline(y=1, color='blue', linestyle='--')
    g.set_ylabel('')
    sns.set_style("white")
    # 创建折线图
    ax2 = g.twinx()
    sns.lineplot(x='dim', y='TC-GNN', data=df1,  color='limegreen', marker='s', ax=ax2, linewidth=2)
    sns.lineplot(x='dim', y='Magicsphere-tf32',data=df1,  color='cornflowerblue', marker='^', ax=ax2,  linewidth=2)
    sns.lineplot(x='dim', y='Magicsphere-fp16', data=df1,  color='royalblue', marker='o', ax=ax2,  linewidth=2)

    # g.tick_params(labelsize=8)
    sns.despine(left=True, right=True, top=True)
    ax2.set_ylabel('')
    # 显示图形
    plt.savefig('./eva100/kernel/gat/result/gat_kernel-100-' + data + '.png', dpi=800)
    # 清空图形
    plt.clf()


# 绘制IGB-medium
baseline = ['Magicsphere-tf32', 'Magicsphere-fp16']
# mycolor = [ 'lightgreen', 'orange', 'gold']
baseline2 = ['TC-GNN', 'Magicsphere-tf32', 'Magicsphere-fp16']
dataset = ['IGB_medium']

compute_tf32 =[]
compute_fp16 =[]
for data in dataset:
    speedup = dict()
    speedup['dim'] = []
    speedup['speed'] = []
    speedup['baseline'] = []
    # speedup['color'] = []

    compute = dict()
    compute['dim'] = []
    compute['TC-GNN'] = []
    compute['Magicsphere-tf32'] = []
    compute['Magicsphere-fp16'] = []
    for dimN in hidden:
        for base, color in zip(baseline, mycolor):
            speedup['dim'].append(str(dimN))
            if res[base][data][dimN]['time'] == -1:
                speedup['speed'].append(0)
            else :
                speed = round( res['TC-GNN'][data][dimN]['time'] / res[base][data][dimN]['time'],2 )
                speedup['speed'].append(speed)
            speedup['baseline'].append(base)
            # speedup['color'].append(color)
    

    for dimN in hidden:
        compute['dim'].append(str(dimN))
        for base in baseline2:
            if res[base][data][dimN]['compute'] == -1 :
                compute[base].append(0)
            else:
                compute[base].append(round( res[base][data][dimN]['compute'],2 ))

            
            
    # 开始绘图：
    df = pd.DataFrame(speedup)
    df1 = pd.DataFrame(compute)
    # print(df)

    
    # sns.set_style("whitegrid")

    # 对数据按照 'dim' 列进行升序排序
    sns.set_style("darkgrid")
    # 设置背景色为灰白色

    g = sns.barplot(x='dim', y='speed', hue='baseline', data=df, palette=mycolor, linewidth=1, legend=False)
    plt.axhline(y=1, color='limegreen', linestyle='--')
    g.set_ylabel('')
    sns.set_style("white")
    # 创建折线图
    ax2 = g.twinx()
    sns.lineplot(x='dim', y='TC-GNN', data=df1,  color='limegreen', marker='s', ax=ax2, linewidth=2)
    sns.lineplot(x='dim', y='Magicsphere-tf32',data=df1,  color='cornflowerblue', marker='^', ax=ax2,  linewidth=2)
    sns.lineplot(x='dim', y='Magicsphere-fp16', data=df1,  color='royalblue', marker='o', ax=ax2,  linewidth=2)

    # g.tick_params(labelsize=8)
    sns.despine(left=True, right=True, top=True)
    ax2.set_ylabel('')
    # 显示图形
    plt.savefig('./eva100/kernel/gat/result/gat_kernel' + data + '.png', dpi=800)
    # 清空图形
    plt.clf()
    compute_tf32+=compute['Magicsphere-tf32']
    compute_fp16+=compute['Magicsphere-fp16']
print('tf32:')
print(max(compute_tf32))
print( sum(compute_tf32) / len(compute_tf32))
print('fp16:')
print(max(compute_fp16))
print( sum(compute_fp16) / len(compute_fp16))