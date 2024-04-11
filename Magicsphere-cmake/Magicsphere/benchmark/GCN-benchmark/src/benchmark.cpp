#include <torch/extension.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <cublas_v2.h>
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

//FP16-8x1
float spmm_forward_cuda_gcn_v2(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_gcn_v2(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    double * values_ = reinterpret_cast<double *>(values.data<at::Half>()); 
    double * rhs_matrix_ = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    double *d_values; 
	double *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_gcn_v2(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//FP16 - 16x1
float spmm_forward_cuda_gcn_16(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_gcn_16(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/16;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    double * values_ = reinterpret_cast<double *>(values.data<at::Half>()); 
    double * rhs_matrix_ = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    double *d_values; 
	double *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

   
    float spmm_ms_avg = spmm_forward_cuda_gcn_16(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//TF32 - 8x1
float spmm_forward_cuda_gcn_tf32_v2(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_gcn_tf32_v2(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    float * values_ = values.data<float>(); 
    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    float *d_values; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_gcn_tf32_v2(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//TF32 - 16x1
float spmm_forward_cuda_gcn_tf32_16(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_gcn_tf32_16(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/16;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    float * values_ = values.data<float>(); 
    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    float *d_values; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

   
    float spmm_ms_avg = spmm_forward_cuda_gcn_tf32_16(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_v2", &spmm_forward_gcn_v2, "SpMM for FP16");
    m.def("forward_16", &spmm_forward_gcn_16, "SpMM for FP16");

    m.def("forward_tf32_v2", &spmm_forward_gcn_tf32_v2, "SpMM for TF32");
    m.def("forward_tf32_16", &spmm_forward_gcn_tf32_16, "SpMM for TF32");

  }