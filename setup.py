from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='composable-sft',
    version='0.0.1',
    description='Tools for training, composing and using sparse fine-tunings of pre-trained neural networks in PyTorch',
    license="Apache",
    long_description=long_description,
    long_description_context_type='text/markdown',
    url='https://github.com/cambridgeltl/composable-sft',
    author='Alan Ansell',
    author_email='aja63@cam.ac.uk',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    install_requires=[
        'conllu',
        'datasets>=1.8',
        'huggingface-hub>=0.0.17',
        'seqeval',
        'transformers>=4.9',
    ],
    python_requires='>=3.9',
)
