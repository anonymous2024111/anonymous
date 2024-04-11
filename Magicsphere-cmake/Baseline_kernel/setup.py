from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='MBaseline_kernel',
    ext_modules=[
      #  CUDAExtension(
      #       name='GNNAdvisor', 
      #       sources=[
      #       './GNNAdvisor/GNNAdvisor_kernel.cu',
      #       './GNNAdvisor/GNNAdvisor.cpp',
      #       ]
      #    ),
       CUDAExtension(
            #v2 without gemm in GNNAdisor
            name='GNNAdvisor_v2_kernel', 
            sources=[
            './GNNAdvisor_v2/GNNAdvisor_kernel.cu',
            './GNNAdvisor_v2/GNNAdvisor.cpp',
            ]
         ),
      #  CUDAExtension(
      #       name='TCGNN_kernel', 
      #       sources=[
      #       './TCGNN/TCGNN_kernel.cu',
      #       './TCGNN/TCGNN.cpp',
      #       ]
      #    ) ,
      #  CUDAExtension(
      #       name='TCGNN_v2_kernel', 
      #       sources=[
      #       './TCGNN_v2/TCGNN_kernel.cu',
      #       './TCGNN_v2/TCGNN.cpp',
      #       ]
      #    ) ,
        CUDAExtension(
            name='TCGNN_kernel', 
            sources=[
            './TCGNN/TCGNN_kernel.cu',
            './TCGNN/TCGNN.cpp',
            ]
         ) ,
       CUDAExtension(
            name='GESpMM_kernel', 
            sources=[
            './GESpMM/gespmmkernel.cu', 
            './GESpMM/gespmm.cpp',
            ]
         ) ,
       CUDAExtension(
            name='cuSPARSE_kernel', 
            sources=[
            './cuSPARSE/spmm_csr_kernel.cu',
            './cuSPARSE/spmm_csr.cpp',
            ]
         ) ,
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


