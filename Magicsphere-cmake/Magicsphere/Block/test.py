import numpy as np
import torch
import MagicsphereBlock_cmake
row = torch.tensor([0,4,5,6,8,8,9,11,13,16,17,17,17,18,18,18,18,22,23,24,26,26,27,29,31,34,35,35,35,36,36,36,36],dtype=torch.int32)
col = torch.tensor([1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)

rowTensor, colTensor, valueTensor, valueTensor_templete = MagicsphereBlock_cmake.blockProcess8_16_csr(row,col)
print(rowTensor)
print(colTensor)
print(valueTensor)
print(valueTensor_templete)
