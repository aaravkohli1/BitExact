from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='bitexact',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='bitexact._C',
            sources=[
                'src/bindings.cpp',
                'src/ops/normalization/rms_norm.cu',
                'src/ops/matmul/matmul.cu',
                'src/ops/reductions/sum.cu',
                'src/ops/reductions/mean.cu',
                'src/ops/activations/sigmoid.cu',
                'src/ops/reductions/max.cu',
                'src/ops/reductions/min.cu',
                'src/ops/normalization/layer_norm.cu',
                'src/ops/reductions/var.cu',
            ],
            include_dirs=[
                'src',
                torch.utils.cpp_extension.include_paths()[0],
            ],
             extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-lineinfo',
                    '-Xptxas', '-O3',
                    '--fmad=false', 
                    '--prec-div=true',     
                    '--prec-sqrt=true',  
                    '-Xcompiler', '-Wall',
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)