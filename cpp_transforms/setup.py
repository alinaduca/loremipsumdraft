from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='cpp_transforms',
    ext_modules=[
        cpp_extension.CppExtension(
            name='cpp_transforms',
            sources=[
                'bindings.cpp',
                'transforms.cpp'
            ],
            include_dirs=['.'],
        ),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
