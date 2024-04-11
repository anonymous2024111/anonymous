#include <torch/extension.h>
#include <vector>
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

torch::Tensor SAG_cuda(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
);


float spmm_forward_cuda(
    float * input,
    float * output,
    int * row_pointers,
    int * column_index,
    float * degrees,
    int *part_pointers,
    int * part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int epoches,
    int dim,
    int num_nodes,
    int num_parts
);

std::vector<torch::Tensor> spmm_backward_cuda(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  );

std::vector<torch::Tensor> spmm_forward_cuda_gin(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  );

std::vector<torch::Tensor> spmm_backward_cuda_gin(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  );

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor SAG(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock) 
{
  CHECK_INPUT(input);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return SAG_cuda(input, row_pointers, column_index, 
              degrees, part_pointers, part2Node, 
              partSize, dimWorker, warpPerBlock);
}


std::vector<torch::Tensor> spmm_forward(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int epoches) 
{
  auto output = torch::zeros({input.size(0), input.size(1)}, torch::kFloat32).to(torch::kCPU);
  // CHECK_INPUT(input);
  // CHECK_INPUT(row_pointers);
  // CHECK_INPUT(column_index);
  // CHECK_INPUT(degrees);
  // CHECK_INPUT(part_pointers);
  // CHECK_INPUT(part2Node);
  float * input_ = input.data<float>(); 
  float * degrees_ = degrees.data<float>(); 
  float * output_ = output.data<float>(); 
  int * row_pointers_ = row_pointers.data<int>();
  int * column_index_ = column_index.data<int>();
  int * part_pointers_ = part_pointers.data<int>();
  int * part2Node_ = part2Node.data<int>();

  int *d_row_pointers, *d_column_index, *d_part_pointers, *d_part2Node;
  float *d_input; 
	float *d_degrees;
  float *d_output;


  checkCuda(cudaMalloc(&d_row_pointers, (row_pointers.size(0)) * sizeof(int)));
  checkCuda(cudaMalloc(&d_column_index, (column_index.size(0)) * sizeof(int)));
  checkCuda(cudaMalloc(&d_part_pointers, (part_pointers.size(0)) * sizeof(int)));
  checkCuda(cudaMalloc(&d_part2Node, (part2Node.size(0)) * sizeof(int)));
  checkCuda(cudaMalloc(&d_input, (input.size(0), input.size(1)) * sizeof(float)));
  checkCuda(cudaMalloc(&d_degrees, (degrees.size(0)) * sizeof(float)));
  checkCuda(cudaMalloc(&d_output, (input.size(0), input.size(1)) * sizeof(float)));

  checkCuda(cudaMemcpy(d_row_pointers, row_pointers_ , (row_pointers.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_column_index, column_index_, (column_index.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_part_pointers, part_pointers_ , (part_pointers.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_part2Node, part2Node_, (part2Node.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_input, input_, (input.size(0), input.size(1)) * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_degrees, degrees_, (degrees.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
  
  float spmm_ms_avg =  spmm_forward_cuda(d_input, d_output, d_row_pointers, d_column_index, 
                            d_degrees, d_part_pointers, d_part2Node, 
                            partSize, dimWorker, warpPerBlock, epoches, output.size(1), output.size(0),  part2Node.size(0));

  checkCuda(cudaMemcpy(output_, d_output, (degrees.size(0)) * sizeof(float), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_pointers);
    cudaFree(d_column_index);
    cudaFree(d_part_pointers);
    cudaFree(d_part2Node);
    cudaFree(d_input);
    cudaFree(d_degrees);
    cudaFree(d_output);
    // delete output_value_cuda;
    return {output, torch::tensor(spmm_ms_avg)};
}

std::vector<torch::Tensor> spmm_backward(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  ) {

  CHECK_INPUT(d_output);
  CHECK_INPUT(X);
  CHECK_INPUT(W);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_backward_cuda(d_output, X, W, row_pointers, column_index, 
                            degrees, part_pointers, part2Node,
                            partSize, dimWorker, warpPerBlock);
}


////////////////////////////////
// spmm forward GIN
///////////////////////////////
std::vector<torch::Tensor> spmm_forward_gin(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_forward_cuda_gin(input, weight, row_pointers, column_index, 
                              epsilon, part_pointers, part2Node, 
                              partSize, dimWorker, warpPerBlock);
}

////////////////////////////////
// spmm backward GIN
///////////////////////////////
std::vector<torch::Tensor> spmm_backward_gin(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  ) {
  CHECK_INPUT(d_output);
  CHECK_INPUT(X);
  CHECK_INPUT(W);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_backward_cuda_gin(d_output, X, W, row_pointers, column_index, 
                            epsilon, part_pointers, part2Node,
                            partSize, dimWorker, warpPerBlock);
}


std::vector<torch::Tensor> build_part(
    int partSize, 
    torch::Tensor indptr
  ) {

  auto indptr_acc = indptr.accessor<int, 1>();
  int num_nodes = indptr.size(0) - 1;
  int degree, thisNumParts, numParts = 0;

	for(int i = 0; i < num_nodes; i++)
	{
    degree = indptr_acc[i + 1] - indptr_acc[i];
	  if(degree % partSize == 0)
			thisNumParts = degree / partSize;
    else
			thisNumParts = degree / partSize + 1;
    numParts += thisNumParts;
	}

  auto partPtr = torch::zeros(numParts + 1);
  auto part2Node = torch::zeros(numParts);
	
  int part_counter = 0;
	for(int i = 0; i < num_nodes; i++)
	{
    int degree = indptr_acc[i + 1] - indptr_acc[i];
    if(degree % partSize == 0)
			thisNumParts = degree / partSize ;
    else
			thisNumParts = degree / partSize + 1;

    for (int pid = 0; pid < thisNumParts; pid++){
      int partBeg = indptr_acc[i] + pid * partSize;
      int partEnd = partBeg + partSize < indptr_acc[i  + 1]? partBeg + partSize: indptr_acc[i + 1];
      partPtr[part_counter] = partBeg;
      part2Node[part_counter++] = i;
      if (i == num_nodes - 1 &&  partEnd == indptr_acc[i + 1])
        partPtr[part_counter] = partEnd;
    }
	}
  return {partPtr, part2Node};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SAG", &SAG, "GNNAdvisor base Scatter-and-Gather Kernel (CUDA)");

  m.def("forward", &spmm_forward, "GNNAdvisor forward (CUDA)");
  m.def("backward", &spmm_backward, "GNNAdvisor backward (CUDA)");

  m.def("forward_gin", &spmm_forward_gin, "GNNAdvisor forward GIN (CUDA)");
  m.def("backward_gin", &spmm_backward_gin, "GNNAdvisor forward GIN (CUDA)");

  m.def("build_part", &build_part, "GNNAdvisor backward (CPU)");
  }