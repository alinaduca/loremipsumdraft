from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='cpp_transforms',
    ext_modules=[
        CppExtension(
            name='cpp_transforms',
            sources=[
                'cpp_transforms/bindings.cpp',
                'cpp_transforms/transforms.cpp',
            ],
            include_dirs=['cpp_transforms'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
