from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='bitexact',
    version='0.1.2',
    author='Aarav Kohli',
    author_email='aaravkohli2008@gmail.com',
    description='Deterministic CUDA operations for reproducible deep learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aaravkohli1/BitExact',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
    ],
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