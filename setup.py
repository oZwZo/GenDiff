from setuptools import setup

setup(
    name = "GenDiff",
    version = '0.1',
    description = 'the python package for diffuse differentiation project. A conditional generative model for single-cell perturbation data',
    author = "Weizhong",
    author_email = "zhengwzh@connect.hku.hk",
    packages = ['src'],
    # package_dir={'': 'src'},
    python_requires='>=3.6'
)