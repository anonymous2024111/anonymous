//
// A demo program of reordering using Rabbit Order.
//
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>

#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>


std::vector<torch::Tensor> rabbit_reorder(
    torch::Tensor in_edge_index, int n, int topK
) ;

std::vector<torch::Tensor> rabbit_reorder_m(
    torch::Tensor in_edge_index, int n, int topK
) ;


std::vector<torch::Tensor> reorderperm(
    torch::Tensor in_edge_index, torch::Tensor com, int n, torch::Tensor mask
);



std::vector<torch::Tensor> rabbit_reorder_tensor(
    torch::Tensor in_edge_index, int n, int topK
) {

  return rabbit_reorder(in_edge_index, n, topK);
}

std::vector<torch::Tensor> rabbit_reorder_tensor_m(
    torch::Tensor in_edge_index, int n, int topK
) {

  return rabbit_reorder_m(in_edge_index, n, topK);
}

std::vector<torch::Tensor> reorderperm_tensor(
    torch::Tensor in_edge_index, torch::Tensor com, int n, torch::Tensor mask
) {
  return reorderperm(in_edge_index, com, n, mask);
}


// binding to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reorder", &rabbit_reorder_tensor, "Get the reordered node id mapping: old_id --> new_id");
  m.def("reorder_m", &rabbit_reorder_tensor_m, "Get the reordered node id mapping: old_id --> new_id");
  m.def("reorderperm", &reorderperm_tensor, "Get the reordered node id mapping: old_id --> new_id");
}