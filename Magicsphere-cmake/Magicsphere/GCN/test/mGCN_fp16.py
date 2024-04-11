import torch
import MagicsphereBlock_cmake
import MagicsphereGCN_cmake
dimN = 512
row = torch.tensor([0,4,5,6,8,8,9,11,13,16,17,17,17,18,18,18,18,22,23,24,26,26,27,29,31,34,35,35,35,36,36,36,36],dtype=torch.int32)
col = torch.tensor([1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)
d=(row[1:] - row[:-1]).tolist()
degree = torch.tensor(d,dtype=torch.half)
# row1,col1,value1 = MagicsphereBlock.blockProcess8_8(row,col,degree)
row1,col1,value1,_ = MagicsphereBlock_cmake.blockProcess8_16_csr(row,col)
values = torch.ones_like(col).half()
# print(torch.reshape(value1, (24, 8)))
rhs = torch.ones((14, dimN),dtype=torch.float16)
# print(rhs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
row1=row1.to(device)
col1=col1.to(device)
value1=value1.to(device)
# values=values.to(device)
# print(row1)
# print(col1)
# print(value1)

rhs=rhs.to(device)
output=MagicsphereGCN_cmake.forward_v2_csr(row1, col1, value1,values, rhs, 16, dimN, 14)
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
