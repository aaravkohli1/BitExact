from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='batchinv',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='batchinv._C',
            sources=[
                'src/bindings.cpp',
                'src/kernels/rms_norm.cu',
            ],
            include_dirs=[
                'src',
                torch.utils.cpp_extension.include_paths()[0],  # Add torch includes
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)