#include<vector>
#include<iostream>
#include<torch/torch.h>
#include<torch/extension.h>

namespace py=pybind11;
namespace tr=torch;
using namespace tr::indexing;
using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;


tr::Tensor d_tanh(tr::Tensor z) {
  return 1 - z.tanh().pow(2);
}

tr::Tensor d_elu(tr::Tensor z, tr::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

void print(tr::Tensor z){
  z.print();
}

tr::Tensor getLine(tr::Tensor &z, int row){
  auto line = z[row];
  z[0] = 0.;
  for (int i = 0; i < z.size(0); i++){
    std::cout<<line[i]<<std::endl;
  }
  return line;
}

bool In(py::object const see, std::vector<py::int_> const see_list){
  
  for (auto x: see_list){
    if (see == x){
      return true;
    }
  }
  return false;
}


tr::Tensor generate_grad(
  tr::Tensor const imgs, tr::Tensor const filters, tr::Tensor const bias,
  std::vector<py::int_> const kernel, std::vector<py::int_> const stride, std::vector<py::int_> const padding,
  tr::Tensor &filter_grad, tr::Tensor &bias_grad, tr::Tensor const grad
){
  tr::Tensor pad_imgs = tr::constant_pad_nd(imgs, tr::IntArrayRef{padding[1], padding[1], padding[0], padding[0]});
  auto batch = grad.size(0);
  auto f_num = filters.size(0);
  tr::Tensor next_grad = tr::zeros_like(pad_imgs); 
  imgs.print();
  filters.print();
  for(int batch_i = 0;batch_i < batch;batch_i++){
    auto image = pad_imgs[batch_i];
    auto grad_i = grad[batch_i];
    auto h = image.size(-2);
    auto w = image.size(-1);
    // std::cout<<h<<' '<<w<<std::endl;
    for(int32_t i = 0;i<h;i++){
      for(int32_t j = 0;j<w;j++){
        if( (i + int(kernel[0]) > h)||(j + int(kernel[1]) > w)){
          continue;
        }
        auto region = image.index({
          "...", Slice(i, i + int(kernel[0])), Slice(j, j+int(kernel[1]))
        });
        int32_t i_stride = i/int(stride[0]);
        int32_t j_stride = j/int(stride[1]);
        for(int32_t f = 0;f < f_num;f++){
          filter_grad[f] += grad_i[f][i_stride][j_stride]*region;
          bias_grad[f]  += grad_i[f][i_stride][j_stride];
          // next_grad[batch_i][][]
          next_grad.index_put_(
            {batch_i, Slice(None), Slice(i, i + int(kernel[0])), Slice(j, j+int(kernel[1]))}, 
            grad_i[f][i_stride][j_stride]*filters[f] + next_grad.index({batch_i, Slice(None), Slice(i, i + int(kernel[0])), Slice(j, j+int(kernel[1]))})
          );
        }
      }
    }
  }
  return next_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("d_tanh", &d_tanh, "d_tanh operator in CPP");
    m.def("d_elu", &d_elu, "d_elu operator in CPP");
    m.def("print", &print, "print information");
    m.def("line", &getLine, "get tensor sizes");
    m.def("In", &In, "in or not");
    m.def("generate_grad", &generate_grad, "Design for Conv2d operator\nGenerate gradients of convolution layer");
}