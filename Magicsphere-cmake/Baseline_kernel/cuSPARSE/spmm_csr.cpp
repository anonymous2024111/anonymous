#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

std::vector<torch::Tensor> cuSPARSE_spmm_csr(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM,
    const long dimN,
    const long nnz,
    int epoches,
    int warmup);


std::vector<torch::Tensor> cuSPARSE_spmm_csr_(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM,
    const long dimN,
    const long nnz,
    int epoches,
    int warmup){
    
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    return cuSPARSE_spmm_csr(row_offsets, col_indices, values, rhs_matrix, dimM, dimN, nnz, epoches, warmup);
}

std::vector<torch::Tensor> cuSPARSE_sddmm_csr(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    const long dimM,
    const long dimN,
    const long nnz,
    int epoches,
    int warmup);

std::vector<torch::Tensor> cuSPARSE_sddmm_csr_(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor lhs_matrix,
    // torch::Tensor rhs_matrix,
    const long dimM,
    const long dimN,
    const long nnz,
    int epoches,
    int warmup){
    
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(lhs_matrix);
    // CHECK_CUDA(rhs_matrix);

    return cuSPARSE_sddmm_csr(row_offsets, col_indices, lhs_matrix, lhs_matrix, dimM, dimN, nnz, epoches, warmup);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuSPARSE_SPMM_CSR", &cuSPARSE_spmm_csr_, "cuSPARSE_SPMM_CSR");
  m.def("cuSPARSE_SDDMM_CSR", &cuSPARSE_sddmm_csr_, "cuSPARSE_SDDMM_CSR");
}