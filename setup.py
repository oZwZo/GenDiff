from setuptools import setup, find_packages

setup(
    name="GenDiff",
    version="0.1",
    description="A conditional generative (diffusion) model for single-cell perturbation data: "
                "learns a displacement field over the differentiation landscape.",
    author="Weizhong",
    author_email="zhengwzh@connect.hku.hk",
    packages=find_packages(include=["gendiff*", "gendiff_dev*", "src*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "anndata",
        "scanpy",
        "torch",
        "pytorch-lightning",
        "matplotlib",
    ],
)
