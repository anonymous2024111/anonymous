#include <torch/extension.h>


std::vector<torch::Tensor> blockProcess8_8(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1);

std::vector<torch::Tensor> blockProcess16_8(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1);

std::vector<torch::Tensor> blockProcess8_4(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1);

std::vector<torch::Tensor> blockProcess16_4(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1);

std::vector<torch::Tensor> blockProcess8_16(torch::Tensor row1, torch::Tensor column1);
std::vector<torch::Tensor> blockProcess8_16_csr(torch::Tensor row1, torch::Tensor column1);

std::vector<torch::Tensor> blockProcess8_16_tf32(torch::Tensor row1, torch::Tensor column1);

std::vector<torch::Tensor> blockProcess_output_8_8(torch::Tensor row1, torch::Tensor column1);

std::vector<torch::Tensor> blockProcess_output_8_4(torch::Tensor row1, torch::Tensor column1);

std::vector<torch::Tensor> blockProcess8_8_1(torch::Tensor row1, torch::Tensor column1);

std::vector<torch::Tensor> blockProcess8_4_1(torch::Tensor row1, torch::Tensor column1);

int blockProcess_tc(torch::Tensor row1, torch::Tensor column1, int m, int n);

int blockProcess_nbs(torch::Tensor row1, torch::Tensor column1, int m, int n);
