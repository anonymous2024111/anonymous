import torch
import MagicsphereBlock
import MagicsphereGCN

# col = torch.tensor([1,3,11,1,7,1,2,5,5,7,0,0,3,8,9,11,1,6,2,7,3,6],dtype=torch.int32)
# row = torch.tensor([0,3,5,6,8,8,8,10,11,14,16,18,20,20,22,22,22],dtype=torch.int32)
row = torch.tensor([0,4,5,6,8,8,9,11,13,16,17,17,17,18,18,18,18,22,23,24,26,26,27,29,31,34,35,35,35,36,36,36,36],dtype=torch.int32)
col = torch.tensor([1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)
d=(row[1:] - row[:-1]).tolist()
degree = torch.tensor(d,dtype=torch.float32)
row1,col1,value1 = MagicsphereBlock.blockProcess8_4(row,col,degree)
# print(torch.reshape(value1, (24, 8)))
# rhs = torch.ones(14, 21)
rhs = torch.ones(30, 20)
# print(rhs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
row1=row1.to(device)
col1=col1.to(device)
value1=value1.to(device)
print(row1)
print(col1)
print(value1)

rhs=rhs.to(device)
# output=MagicsphereGCN.forward_tf32(row1, col1, value1, rhs, 16, 21, 14)
output=MagicsphereGCN.forward_tf32_v2(row1, col1, value1, rhs, 32, 20, 30)
# outputcpu=value1.to('cpu').numpy()
# for i in range(16):
#     for j in range(8):
#         print(outputcpu[ i*8 + j], end=" ")
#     print()
#     if i%8==7 :
#         print()
print(output)
print(output.shape)
# non_zero_tensor = output[output != 0]
# print(non_zero_tensor)

# result = example.add(3, 4)
# print(result)  # 输出结果为7
