from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='MBaseline_cmake',
    ext_modules=[
       CUDAExtension(
            name='GNNAdvisor_cmake', 
            sources=[
            './GNNAdvisor/GNNAdvisor_kernel.cu',
            './GNNAdvisor/GNNAdvisor.cpp',
            ]
         ),
       CUDAExtension(
            #v2 without gemm in GNNAdisor
            name='GNNAdvisor_v2_cmake', 
            sources=[
            './GNNAdvisor_v2/GNNAdvisor_kernel.cu',
            './GNNAdvisor_v2/GNNAdvisor.cpp',
            ]
         ),
       CUDAExtension(
            name='TCGNN_cmake', 
            sources=[
            './TCGNN/TCGNN_kernel.cu',
            './TCGNN/TCGNN.cpp',
            ]
         ) ,
       CUDAExtension(
            name='TCGNN_v2_cmake', 
            sources=[
            './TCGNN_v2/TCGNN_kernel.cu',
            './TCGNN_v2/TCGNN.cpp',
            ]
         ) ,
       CUDAExtension(
            name='GESpMM_cmake', 
            sources=[
            './GESpMM/gespmmkernel.cu', 
            './GESpMM/gespmm.cpp',
            ]
         ) ,
       CUDAExtension(
            name='cuSPARSE_cmake', 
            sources=[
            './cuSPARSE/spmm_csr_kernel.cu',
            './cuSPARSE/spmm_csr.cpp',
            ]
         ) ,
       CppExtension(
            name='Rabbit_cmake', 
            sources=[
            './Rabbit/reorder.cpp',
            ],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            libraries=["numa", "tcmalloc_minimal"]
         )  
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


