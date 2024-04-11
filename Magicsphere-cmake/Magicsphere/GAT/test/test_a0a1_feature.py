import torch
import MagicsphereGAT_cmake
import MagicsphereBlock_cmake
row = torch.tensor([0,4,5,6,8,8,9,11,13,16,17,17,17,18,18,18,18,22,23,24,26,26,27,29,31,34,35,35,35,36,36,36,36],dtype=torch.int32)
col = torch.tensor([1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rowTensor, colTensor, valueTensor, valueTensor_templete = MagicsphereBlock_cmake.blockProcess8_16_csr(row,col)
print(rowTensor)
print(colTensor)
print(valueTensor)
print(valueTensor_templete)
rowTensor = rowTensor.to(device)
colTensor = colTensor.to(device)
valueTensor = valueTensor.to(device)

dimM = 32
dimMori=30
dimN = 5
#rhs = torch.ones((dimMori, dimN),dtype=torch.float16)
# a0 = torch.ones((dimN),dtype=torch.float16)
# a1 = torch.ones((dimN),dtype=torch.float16)*2

a0 = torch.randint(low=1, high=4, size=(dimN,))
a1 = torch.randint(low=1, high=4, size=(dimN,))
rhs = torch.randint(low=1, high=4, size=(dimMori, dimN))
a0 = a0.half()
a1 = a1.half()
rhs = rhs.half()

rhs=rhs.to(device)
a0=a0.to(device)
a1=a1.to(device)

output_a0, output_a1=MagicsphereGAT_cmake.fp16_a_feature(dimN, dimM, rhs, a0, a1)
print(output_a0)
print(output_a1)
result = rowTensor[2::2] - rowTensor[:-2:2]
max_vectors=max(result)
attention = MagicsphereGAT_cmake.fp16_csr(dimN, dimM, rowTensor, colTensor, valueTensor,output_a0, output_a1, max_vectors.item(), col.size(0))
print(attention)
print()