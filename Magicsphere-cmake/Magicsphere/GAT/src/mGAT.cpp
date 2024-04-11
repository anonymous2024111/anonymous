#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")


/*
FP16
*/
torch::Tensor sddmm_forward_cuda_gat(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors);

torch::Tensor sddmm_forward_gat(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    CHECK_CUDA(weights0);
    CHECK_CUDA(weights1);

    return sddmm_forward_cuda_gat(dimN, dimM,
    row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    weights0,
    weights1,
    max_vectors);    
}


//csr
torch::Tensor sddmm_forward_cuda_gat_csr(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors,
    int nnz);

torch::Tensor sddmm_forward_gat_csr(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors,
    int nnz)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    CHECK_CUDA(weights0);
    CHECK_CUDA(weights1);

    return sddmm_forward_cuda_gat_csr(dimN, dimM,
    row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    weights0,
    weights1,
    max_vectors,
    nnz);    
}

torch::Tensor sddmm_gen_forward_cuda_gat(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int max_vectors);

torch::Tensor sddmm_gen_forward_gat(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int max_vectors)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(lhs_matrix);
    CHECK_CUDA(rhs_matrix);

    return sddmm_gen_forward_cuda_gat(dimN, dimM,
    row_offsets,
    col_indices, 
    values, 
    lhs_matrix,
    rhs_matrix,
    max_vectors);    

}

torch::Tensor trans_cuda_gat(
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor values_templete,
    int max_vectors);

torch::Tensor trans_gat(
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor values_templete,
    int max_vectors)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(values_templete);

    return trans_cuda_gat(dimM,
    row_offsets,
    col_indices, 
    values, 
    values_templete,
    max_vectors);    

}


/*
TF32
*/
torch::Tensor sddmm_forward_cuda_gat_tf32(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors);

torch::Tensor sddmm_forward_gat_tf32(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    CHECK_CUDA(weights0);
    CHECK_CUDA(weights1);

    return sddmm_forward_cuda_gat_tf32(dimN, dimM,
    row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    weights0,
    weights1,
    max_vectors);    
}


//csr
torch::Tensor sddmm_forward_cuda_gat_tf32_csr(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors,
    int nnz);

torch::Tensor sddmm_forward_gat_tf32_csr(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors,
    int nnz)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    CHECK_CUDA(weights0);
    CHECK_CUDA(weights1);

    return sddmm_forward_cuda_gat_tf32_csr(dimN, dimM,
    row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    weights0,
    weights1,
    max_vectors,
    nnz);    
}

torch::Tensor sddmm_gen_forward_cuda_gat_tf32(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int max_vectors);

torch::Tensor sddmm_gen_forward_gat_tf32(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int max_vectors)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(lhs_matrix);
    CHECK_CUDA(rhs_matrix);

    return sddmm_gen_forward_cuda_gat_tf32(dimN, dimM,
    row_offsets,
    col_indices, 
    values, 
    lhs_matrix,
    rhs_matrix,
    max_vectors);    

}

torch::Tensor trans_cuda_gat_tf32(
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor values_templete,
    int max_vectors);

torch::Tensor trans_gat_tf32(
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor values_templete,
    int max_vectors)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(values_templete);

    return trans_cuda_gat_tf32(dimM,
    row_offsets,
    col_indices, 
    values, 
    values_templete,
    max_vectors);    

}



//fp16 forward trans
std::vector<torch::Tensor> sddmm_forward_cuda_gat_trans(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors);
    
std::vector<torch::Tensor> sddmm_forward_gat_trans(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    torch::Tensor weights0,
    torch::Tensor weights1,
    int max_vectors)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    CHECK_CUDA(weights0);
    CHECK_CUDA(weights1);

    return sddmm_forward_cuda_gat_trans(dimN, dimM,
    row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    weights0,
    weights1,
    max_vectors);    
}

std::vector<torch::Tensor> gat_fp16_a_feature_cuda(
    const long dimN, 
    const long dimM,
    torch::Tensor rhs_matrix,
    torch::Tensor a0,
    torch::Tensor a1);
std::vector<torch::Tensor> gat_fp16_a_feature(
    const long dimN, 
    const long dimM1,
    torch::Tensor rhs_matrix,
    torch::Tensor a0,
    torch::Tensor a1)
{
    int dimM=dimM1/16;
    CHECK_CUDA(rhs_matrix);
    CHECK_CUDA(a0);
    CHECK_CUDA(a1);

    return gat_fp16_a_feature_cuda(
    dimN,
    dimM, 
    rhs_matrix, 
    a0,
    a1);    
}

// a0, a1 concat
torch::Tensor gat_fp16_csr_cuda(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor a0,
    torch::Tensor a1,
    int max_vectors,
    int nnz);
torch::Tensor gat_fp16_csr(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor a0,
    torch::Tensor a1,
    int max_vectors,
    int nnz)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(a0);
    CHECK_CUDA(a1);

    return gat_fp16_csr_cuda(
    dimN,
    dimM, 
    row_offsets,
    col_indices,
    values,
    a0,
    a1,
    max_vectors,
    nnz);
}


torch::Tensor gat_fp16_sddmm_csr_cuda(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int max_vectors,
    int nnz);

torch::Tensor gat_fp16_sddmm_csr(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int max_vectors,
    int nnz)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(lhs_matrix);
    CHECK_CUDA(rhs_matrix);

    return gat_fp16_sddmm_csr_cuda(dimN, dimM,
    row_offsets,
    col_indices, 
    values, 
    lhs_matrix,
    rhs_matrix,
    max_vectors,
    nnz);    

}

// tf32 csr

torch::Tensor gat_tf32_sddmm_csr_cuda(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int max_vectors,
    int nnz);

torch::Tensor gat_tf32_sddmm_csr(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int max_vectors,
    int nnz)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(lhs_matrix);
    CHECK_CUDA(rhs_matrix);

    return gat_tf32_sddmm_csr_cuda(dimN, dimM,
    row_offsets,
    col_indices, 
    values, 
    lhs_matrix,
    rhs_matrix,
    max_vectors,
    nnz);    

}


std::vector<torch::Tensor> gat_tf32_a_feature_cuda(
    const long dimN, 
    const long dimM,
    torch::Tensor rhs_matrix,
    torch::Tensor a0,
    torch::Tensor a1);
std::vector<torch::Tensor> gat_tf32_a_feature(
    const long dimN, 
    const long dimM1,
    torch::Tensor rhs_matrix,
    torch::Tensor a0,
    torch::Tensor a1)
{
    int dimM=dimM1/16;
    CHECK_CUDA(rhs_matrix);
    CHECK_CUDA(a0);
    CHECK_CUDA(a1);

    return gat_tf32_a_feature_cuda(
    dimN,
    dimM, 
    rhs_matrix, 
    a0,
    a1);    
}

// a0, a1 concat
torch::Tensor gat_tf32_csr_cuda(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor a0,
    torch::Tensor a1,
    int max_vectors,
    int nnz);
torch::Tensor gat_tf32_csr(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor a0,
    torch::Tensor a1,
    int max_vectors,
    int nnz)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(a0);
    CHECK_CUDA(a1);

    return gat_tf32_csr_cuda(
    dimN,
    dimM, 
    row_offsets,
    col_indices,
    values,
    a0,
    a1,
    max_vectors,
    nnz);
}

// a0, a1 concat
torch::Tensor gat_tf32_csr_cuda_v2(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor a0,
    torch::Tensor a1,
    int nnz);
torch::Tensor gat_tf32_csr_v2(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor a0,
    torch::Tensor a1,
    int nnz)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(a0);
    CHECK_CUDA(a1);

    return gat_tf32_csr_cuda_v2(
    dimN,
    dimM, 
    row_offsets,
    col_indices,
    a0,
    a1,
    nnz);
}

torch::Tensor gat_fp16_csr_cuda_v2(
    const long dimN, 
    const long dimM,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor a0,
    torch::Tensor a1,
    int nnz);
torch::Tensor gat_fp16_csr_v2(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor a0,
    torch::Tensor a1,
    int nnz)
{
    int dimM=dimM1/8;
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(a0);
    CHECK_CUDA(a1);

    return gat_fp16_csr_cuda_v2(
    dimN,
    dimM, 
    row_offsets,
    col_indices,
    a0,
    a1,
    nnz);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sddmm_forward_gat, "sddmm FP16 for GAT, output is 8x16 and save as 8x8");
  m.def("forward_csr", &sddmm_forward_gat_csr, "sddmm FP16 for GAT, output is 8x16 and save as 8x8");
  m.def("forward_gen", &sddmm_gen_forward_gat, "general sddmm FP16");
  m.def("trans_gat", &trans_gat, "trans GAT for FP16 and save as 8x8");
  
  m.def("forward_tf32", &sddmm_forward_gat_tf32, "sddmm TF32 for GAT, output is 8x16 and save as 8x4");
  m.def("forward_tf32_csr", &sddmm_forward_gat_tf32_csr, "sddmm TF32 for GAT, output is 8x16 and save as 8x4");
  m.def("forward_gen_tf32", &sddmm_gen_forward_gat_tf32, "general sddmm TF32");
  m.def("trans_gat_tf32", &trans_gat_tf32, "trans GAT for TF32 and save as 8x4");

  m.def("forward_trans", &sddmm_forward_gat_trans, "sddmm FP16 for GAT, output is 8x16 and save as 8x8");


  m.def("fp16_a_feature", &gat_fp16_a_feature, "compute a0, a1 x feature");
  m.def("fp16_csr", &gat_fp16_csr, "compute a0, a1 x feature");
  m.def("fp16_csr_v2", &gat_fp16_csr_v2, "compute a0, a1 x feature");
  m.def("fp16_sddmm_csr", &gat_fp16_sddmm_csr, "sddmm csr");

  m.def("tf32_a_feature", &gat_tf32_a_feature, "compute a0, a1 x feature");
  m.def("tf32_csr", &gat_tf32_csr, "compute a0, a1 x feature");
  m.def("tf32_csr_v2", &gat_tf32_csr_v2, "compute a0, a1 x feature");
  m.def("tf32_sddmm_csr", &gat_tf32_sddmm_csr, "sddmm csr");
    }