from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='Magicsphere_kernel',
    # ext_modules=[module],
    ext_modules=[
       CUDAExtension(
            name='MagicsphereGCN_kernel', 
            sources=[
            './GCN-benchmark/src/benchmark.cpp'
            ],
            library_dirs=['./GCN-benchmark/lib/libMGCN1.a'], 
            extra_objects=['./GCN-benchmark/lib/libMGCN1.a'],
            extra_compile_args=['-O3']
         ),
       CUDAExtension(
            name='MagicsphereGAT_kernel', 
            sources=[
            './GAT-benchmark/src/benchmark.cpp'
            ],
            library_dirs=['./GAT-benchmark/lib/libMGAT1.a'], 
            extra_objects=['./GAT-benchmark/lib/libMGAT1.a'],
            extra_compile_args=['-O3']
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


