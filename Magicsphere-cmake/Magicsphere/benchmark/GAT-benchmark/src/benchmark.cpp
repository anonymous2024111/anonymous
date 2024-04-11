#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

inline
cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

/*
FP16
*/

float sddmm_gen_forward_cuda_gat(
    long dimN, 
    long dimM, 
    int * row_offsets, 
    int * col_indices,
    half * values,
    half * lhs_matrix,
    half * rhs_matrix,
    half * output_matrix,
    int max_vectors,
    int dimMori,
    int epoches,
    int warps);

std::vector<torch::Tensor> sddmm_gen_forward_gat(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    // torch::Tensor rhs_matrix,
    int max_vectors,
    int epoches,
    int warps)
{
    int dimM=dimM1/8;
    int mOri = lhs_matrix.size(0);
    auto output_matrix = torch::zeros({values.size(0)}, torch::kFloat16).to(torch::kCPU);

    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    half * values_ = reinterpret_cast< half*>(values.data<at::Half>()); 
    half * lhs_matrix_ = reinterpret_cast< half*>(lhs_matrix.data<at::Half>()); 
    // half * rhs_matrix_ = reinterpret_cast< half*>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast< half*>(output_matrix.data<at::Half>());

    int *d_row_offsets, *d_col_indices;
    half *d_values; 
	half *d_lhs_matrix;
    // half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_lhs_matrix, (mOri*dimN) * sizeof(half)));
    // checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (values.size(0)) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    
    float spmm_ms_avg = sddmm_gen_forward_cuda_gat(dimN, dimM,
    d_row_offsets,
    d_col_indices, 
    d_values, 
    d_lhs_matrix,
    d_lhs_matrix,
    d_output_matrix,
    max_vectors,
    mOri,
    epoches,
    warps);    

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (values.size(0))* sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_lhs_matrix);
    // cudaFree(d_lhs_matrix);
    cudaFree(d_output_matrix);
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//16x1, m16n8k16
float sddmm_gen_forward_cuda_gat_16(
    long dimN, 
    long dimM, 
    int * row_offsets, 
    int * col_indices,
    half * values,
    half * lhs_matrix,
    half * rhs_matrix,
    half * output_matrix,
    int max_vectors,
    int dimMori,
    int epoches,
    int warps);

std::vector<torch::Tensor> sddmm_gen_forward_gat_16(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    // torch::Tensor rhs_matrix,
    int max_vectors,
    int epoches,
    int warps)
{
    int dimM=dimM1/16;
    int mOri = lhs_matrix.size(0);
    auto output_matrix = torch::zeros({values.size(0)}, torch::kFloat16).to(torch::kCPU);

    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    half * values_ = reinterpret_cast< half*>(values.data<at::Half>()); 
    half * lhs_matrix_ = reinterpret_cast< half*>(lhs_matrix.data<at::Half>()); 
    // half * rhs_matrix_ = reinterpret_cast< half*>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast< half*>(output_matrix.data<at::Half>());

    int *d_row_offsets, *d_col_indices;
    half *d_values; 
	half *d_lhs_matrix;
    // half *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_lhs_matrix, (mOri*dimN) * sizeof(half)));
    // checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (values.size(0)) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    
    float spmm_ms_avg = sddmm_gen_forward_cuda_gat_16(dimN, dimM,
    d_row_offsets,
    d_col_indices, 
    d_values, 
    d_lhs_matrix,
    d_lhs_matrix,
    d_output_matrix,
    max_vectors,
    mOri,
    epoches,
    warps);    

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (values.size(0)) * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_lhs_matrix);
    // cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


/*
TF32
*/

float sddmm_gen_forward_cuda_gat_tf32(
    const long dimN, 
    const long dimM,
    int * row_offsets, 
    int * col_indices,
    float *  values,
    float *  lhs_matrix,
    float *  rhs_matrix,
    float * output_matrix,
    int max_vectors,
    int dimMori,
    int epoches,
    int warps);

std::vector<torch::Tensor> sddmm_gen_forward_gat_tf32(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    // torch::Tensor rhs_matrix,
    int max_vectors,
    int epoches,
    int warps)
{
    int dimM=dimM1/8;
    int mOri = lhs_matrix.size(0);
    auto output_matrix = torch::zeros({values.size(0)}, torch::kFloat32).to(torch::kCPU);

    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    float * values_ = values.data<float>(); 
    float * lhs_matrix_ = lhs_matrix.data<float>(); 
    // float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    int *d_row_offsets, *d_col_indices;
    float *d_values; 
	float *d_lhs_matrix;
    // float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_lhs_matrix, (mOri*dimN) * sizeof(float)));
    // checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (values.size(0)) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg = sddmm_gen_forward_cuda_gat_tf32(dimN, dimM,
    d_row_offsets,
    d_col_indices, 
    d_values, 
    d_lhs_matrix,
    d_lhs_matrix,
    d_output_matrix,
    max_vectors,
    mOri,
    epoches,
    warps);   

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (values.size(0)) * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_lhs_matrix);
    // cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

// 16x1 m16n8k8
float sddmm_gen_forward_cuda_gat_tf32_16(
    const long dimN, 
    const long dimM,
    int * row_offsets, 
    int * col_indices,
    float *  values,
    float *  lhs_matrix,
    float *  rhs_matrix,
    float * output_matrix,
    int max_vectors,
    int dimMori,
    int epoches,
    int warps);

std::vector<torch::Tensor> sddmm_gen_forward_gat_tf32_16(
    const long dimN, 
    const long dimM1,
    torch::Tensor row_offsets, 
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor lhs_matrix,
    // torch::Tensor rhs_matrix,
    int max_vectors,
    int epoches,
    int warps)
{
    int dimM=dimM1/16;
    int mOri = lhs_matrix.size(0);
    auto output_matrix = torch::zeros({values.size(0)}, torch::kFloat32).to(torch::kCPU);

    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    float * values_ = values.data<float>(); 
    float * lhs_matrix_ = lhs_matrix.data<float>(); 
    // float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

    int *d_row_offsets, *d_col_indices;
    float *d_values; 
	float *d_lhs_matrix;
    // float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_lhs_matrix, (mOri*dimN) * sizeof(float)));
    // checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (values.size(0)) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));
    // checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    float spmm_ms_avg = sddmm_gen_forward_cuda_gat_tf32_16(dimN, dimM,
    d_row_offsets,
    d_col_indices, 
    d_values, 
    d_lhs_matrix,
    d_lhs_matrix,
    d_output_matrix,
    max_vectors,
    mOri,
    epoches,
    warps);   

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, (values.size(0)) * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_lhs_matrix);
    // cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_gen", &sddmm_gen_forward_gat, "general sddmm FP16");
  m.def("forward_gen_16", &sddmm_gen_forward_gat_16, "general sddmm FP16");

  m.def("forward_gen_tf32", &sddmm_gen_forward_gat_tf32, "general sddmm TF32");
  m.def("forward_gen_tf32_16", &sddmm_gen_forward_gat_tf32_16, "general sddmm TF32");

    }