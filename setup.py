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
                'src/ops/reductions/mean.cu'
            ],
            include_dirs=[
                'src',
                torch.utils.cpp_extension.include_paths()[0],  # Add torch includes
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)