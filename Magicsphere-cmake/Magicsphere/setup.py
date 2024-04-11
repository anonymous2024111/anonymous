from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


#创建 Extension 对象
# module = CUDAExtension('MagicsphereGCN1',
#                    sources=['./GCN/src/mGCN.cpp'],  # 空列表表示没有需要编译的额外源文件
#                    libraries=['./GCN/lib/libMGCN'],  # 需要链接的动态库名称
#                    library_dirs=['./GCN/lib/libMGCN'],  # 动态库所在的目录
#                    runtime_library_dirs=['./lib/libMGCN'],  # 运行时动态库搜索路径
#                    )
# module = CUDAExtension('MagicsphereGCN1',
#                    sources=['./GCN/src/mGCN.cpp'],  # 空列表表示没有需要编译的额外源文件
#                    extra_objects=['./GCN/lib/libMGCN.a'],
#                    library_dirs=['./GCN/lib/libMGCN.a'] 
#                    )

#设置编译参数
# extra_link_args = ['-Wl,-rpath=./lib/libMGCN']  # 设置运行时动态库搜索路径，可选
# module.extra_link_args = extra_link_args


# setup(
#      name='Magicsphere1',
#      ext_modules=[module])

setup(
    name='Magicsphere_cmake',
    # ext_modules=[module],
    ext_modules=[
      #  CUDAExtension(
      #       name='MagicsphereGCN1', 
      #       sources=[
      #       './GCN/src/mGCN.cpp'
      #       ],
      #       library_dirs=['./GCN/lib/libMGCN.a'], 
      #       extra_objects=['./GCN/lib/libMGCN.a'],
      #       extra_compile_args=['-O3']
      #    ),
       CUDAExtension(
            name='MagicsphereGCN_cmake', 
            sources=[
            './GCN/src/mGCN.cpp'
            ],
            library_dirs=['./GCN/lib/libMGCN.a'], 
            extra_objects=['./GCN/lib/libMGCN.a'],
            extra_compile_args=['-O3']
         ),
       CUDAExtension(
            name='MagicsphereGAT_cmake', 
            sources=[
            './GAT/src/mGAT.cpp',
            ],
            library_dirs=['./GAT/lib/libMGAT.a'], 
            extra_objects=['./GAT/lib/libMGAT.a'],
            extra_compile_args=['-O3']
         ) ,
       CUDAExtension(
            name='MagicsphereBlock_cmake', 
            sources=[
            './Block/example.cpp'
            ],
            library_dirs=['./Block/lib/libMBlock.a'], 
            extra_objects=['./Block/lib/libMBlock.a'],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
         ) ,
       CppExtension(
            name='MagicsphereMRabbit_cmake', 
            sources=[
            './LRabbitAll2/reorder.cpp',
            ],
            library_dirs=['.LRabbitAll2/lib/libMRabbitAll.a'], 
            extra_objects=['./LRabbitAll2/lib/libMRabbitAll.a'],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            libraries=["numa", "tcmalloc_minimal"]
         )  
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


