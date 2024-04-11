#include <torch/extension.h>
#include <cuda_fp16.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

/*
FP16
*/
torch::Tensor spmm_forward_cuda_gcn(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_gcn(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn(row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
}

//fp16 16
torch::Tensor spmm_forward_cuda_gcn_16(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_gcn_16(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/16;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn_16(row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
}
//fp16 ns
torch::Tensor spmm_forward_cuda_gcn_ns(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_gcn_ns(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn_ns(row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
}
/*
TF32
*/

torch::Tensor spmm_forward_cuda_gcn_tf32(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_gcn_tf32(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
   //std::cout<<"tf32-Magicsphere"<<std::endl;
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn_tf32(row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
    // torch::Tensor tensor = torch::tensor({1, 2, 3,4,5,6}, torch::kInt32);
    // return tensor;
}


/*
TF32
*/

torch::Tensor spmm_forward_cuda_gcn_tf32_16(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_gcn_tf32_16(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
   //std::cout<<"tf32-Magicsphere"<<std::endl;
    int dimM=dimM1/16;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn_tf32_16(row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
    // torch::Tensor tensor = torch::tensor({1, 2, 3,4,5,6}, torch::kInt32);
    // return tensor;
}


/*
TF32
*/

torch::Tensor spmm_forward_cuda_gcn_tf32_ns(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_gcn_tf32_ns(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
   //std::cout<<"tf32-Magicsphere"<<std::endl;
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn_tf32_ns(row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
    // torch::Tensor tensor = torch::tensor({1, 2, 3,4,5,6}, torch::kInt32);
    // return tensor;
}



//V2

torch::Tensor spmm_forward_cuda_gcn_v2(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_gcn_v2(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn_v2(row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
}

//sum
torch::Tensor spmm_forward_cuda_sum(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor templete, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_sum(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor templete, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(templete);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_sum(row_offsets,
    col_indices, 
    values, 
    templete,
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
}

// // fp16 softmax
// torch::Tensor spmm_forward_cuda_gcn_v2_softmax(
//     torch::Tensor row_offsets,
//     torch::Tensor col_indices, 
//     torch::Tensor values, 
//     torch::Tensor templete,
//     torch::Tensor rhs_martix,
//     const long dimM,
//     const long dimN,
//     const long mOri);


// torch::Tensor spmm_forward_gcn_v2_softmax(
//     torch::Tensor row_offsets,
//     torch::Tensor col_indices, 
//     torch::Tensor values, 
//     torch::Tensor templete,
//     torch::Tensor rhs_matrix,
//     const long dimM1,
//     const long dimN,
//     const long mOri)
// {
//     int dimM=dimM1/8;
//     CHECK_CUDA(row_offsets);
//     CHECK_CUDA(col_indices);
//     CHECK_CUDA(values);
//     CHECK_CUDA(rhs_matrix);
//     CHECK_CUDA(templete);
//     return spmm_forward_cuda_gcn_v2_softmax(row_offsets,
//     col_indices, 
//     values, 
//     templete,
//     rhs_matrix,
//     dimM,
//     dimN,
//     mOri);    
// }

torch::Tensor spmm_forward_cuda_gcn_tf32_v2(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_gcn_tf32_v2(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
   //std::cout<<"tf32-Magicsphere"<<std::endl;
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn_tf32_v2(row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
    // torch::Tensor tensor = torch::tensor({1, 2, 3,4,5,6}, torch::kInt32);
    // return tensor;
}

torch::Tensor spmm_forward_cuda_gcn_tf32_sum(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor templete, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor spmm_forward_tf32_sum(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor templete, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
   //std::cout<<"tf32-Magicsphere"<<std::endl;
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(templete);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn_tf32_sum(row_offsets,
    col_indices, 
    values, 
    templete,
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
    // torch::Tensor tensor = torch::tensor({1, 2, 3,4,5,6}, torch::kInt32);
    // return tensor;
}



//csr
torch::Tensor gat_fp16_spmm_csr_cuda(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor values_csr, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor gat_fp16_spmm_csr(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor values_csr, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(values_csr);
    CHECK_CUDA(rhs_matrix);
    
    return gat_fp16_spmm_csr_cuda(row_offsets,
    col_indices, 
    values, 
    values_csr,
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
}

//tf32 csr

torch::Tensor gat_tf32_spmm_csr_cuda(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor values_csr, 
    torch::Tensor rhs_martix,
    const long dimM,
    const long dimN,
    const long mOri);


torch::Tensor gat_tf32_spmm_csr(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor values_csr, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
   //std::cout<<"tf32-Magicsphere"<<std::endl;
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(values_csr);
    CHECK_CUDA(rhs_matrix);
    
    return gat_tf32_spmm_csr_cuda(row_offsets,
    col_indices, 
    values, 
    values_csr,
    rhs_matrix,
    dimM,
    dimN,
    mOri);    
    // torch::Tensor tensor = torch::tensor({1, 2, 3,4,5,6}, torch::kInt32);
    // return tensor;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spmm_forward_gcn, "SpMM for FP16");
  m.def("forward_v2", &spmm_forward_gcn_v2, "SpMM for FP16");
  m.def("forward_16", &spmm_forward_gcn_16, "SpMM for FP16");
  m.def("forward_ns", &spmm_forward_gcn_ns, "SpMM for FP16");

  m.def("forward_tf32", &spmm_forward_gcn_tf32, "SpMM for TF32");
  m.def("forward_tf32_v2", &spmm_forward_gcn_tf32_v2, "SpMM for TF32");
  m.def("forward_tf32_16", &spmm_forward_gcn_tf32_16, "SpMM for TF32");
  m.def("forward_tf32_ns", &spmm_forward_gcn_tf32_ns, "SpMM for TF32");

  m.def("forward_filter", &spmm_forward_sum, "SpMM for FP16");
  m.def("forward_tf32_filter", &spmm_forward_tf32_sum, "SpMM for FP16");
//   m.def("forward_v2_softmax", &spmm_forward_gcn_v2_softmax, "SpMM for FP16");

  m.def("forward_v2_csr", &gat_fp16_spmm_csr, "spmm csr");
  m.def("forward_tf32_v2_csr", &gat_tf32_spmm_csr, "spmm csr");
  }