#include <omp.h>
#include "./include/example.h"

std::vector<torch::Tensor> blockProcess8_8_tensor(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1)
{
    return blockProcess8_8(row1,column1,degree1);
}

std::vector<torch::Tensor> blockProcess16_8_tensor(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1)
{
    return blockProcess16_8(row1,column1,degree1);
}

std::vector<torch::Tensor> blockProcess8_4_tensor(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1)
{
   return blockProcess8_4(row1,column1,degree1);
}

std::vector<torch::Tensor> blockProcess16_4_tensor(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1)
{
   return blockProcess16_4(row1,column1,degree1);
}

std::vector<torch::Tensor> blockProcess8_16_tensor(torch::Tensor row1, torch::Tensor column1)
{
    return blockProcess8_16(row1,column1);
}

std::vector<torch::Tensor> blockProcess8_16_csr_tensor(torch::Tensor row1, torch::Tensor column1)
{
    return blockProcess8_16_csr(row1,column1);
}

std::vector<torch::Tensor> blockProcess8_16_tf32_tensor(torch::Tensor row1, torch::Tensor column1)
{
    return blockProcess8_16_tf32(row1,column1);
}

std::vector<torch::Tensor> blockProcess_output_8_8_tensor(torch::Tensor row1, torch::Tensor column1)
{
    return blockProcess_output_8_8(row1,column1);
}

std::vector<torch::Tensor> blockProcess_output_8_4_tensor(torch::Tensor row1, torch::Tensor column1)
{
   return blockProcess_output_8_4(row1,column1);
}

std::vector<torch::Tensor> blockProcess8_8_1_tensor(torch::Tensor row1, torch::Tensor column1)
{
    return blockProcess8_8_1(row1,column1);
}

std::vector<torch::Tensor> blockProcess8_4_1_tensor(torch::Tensor row1, torch::Tensor column1)
{
    return blockProcess8_4_1(row1,column1);
}

int blockProcess_tc_tensor(torch::Tensor row1, torch::Tensor column1, int m, int n)
{
    return blockProcess_tc(row1,column1,m,n);
}

int blockProcess_nbs_tensor(torch::Tensor row1, torch::Tensor column1, int m, int n)
{
    return blockProcess_nbs(row1,column1,m,n);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //MGCN
    m.def("blockProcess8_8", &blockProcess8_8_tensor, "MGCN for FP16 with shape 8x8");
    m.def("blockProcess8_4", &blockProcess8_4_tensor, "MGCN for TF32 with block 8x4");
    m.def("blockProcess16_8", &blockProcess16_8_tensor, "MGCN for FP16 with block 16x4");
    m.def("blockProcess16_4", &blockProcess16_4_tensor, "MGCN for TF32 with block 8x4");

    //MGAT
    m.def("blockProcess8_16", &blockProcess8_16_tensor, "MGAT for FP16 with output 8x16");
    m.def("blockProcess8_16_csr", &blockProcess8_16_csr_tensor, "MGAT for FP16 with output 8x16");
    m.def("blockProcess8_16_tf32", &blockProcess8_16_tf32_tensor, "MGAT for TF32 with output 8x16");
    m.def("blockProcess_output_8_8", &blockProcess_output_8_8_tensor, "MGAT for FP16 with output 8x16 and save as 8x8");
    m.def("blockProcess_output_8_4", &blockProcess_output_8_4_tensor, "MGAT for TF32 with output 8x16 and save as 8x4");

    //Blocks Evaluation
    m.def("blockProcess8_8_1", &blockProcess8_8_1_tensor, "Only 8x8 block");
    m.def("blockProcess8_4_1", &blockProcess8_4_1_tensor, "Only 8x4 block");
    m.def("blockProcess_tc", &blockProcess_tc_tensor, "Only blocks for TC-GNN, SGT");
    m.def("blockProcess_nbs", &blockProcess_nbs_tensor, "Only blocks for NBS");


}