from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='composable-sft',
    version='0.0.1',
    description='Tools for training, composing and using sparse fine-tunings of pre-trained neural networks in PyTorch',
    long_description=long_description,
    long_description_context_type='text/markdown',
    url='https://github.com/cambridgeltl/composable-sft',
    author='Alan Ansell',
    author_email='aja63@cam.ac.uk',
    #classifiers =
    #    Programming Language :: Python :: 3

    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    install_requires=[
        'transformers>=4.9',
    ],
    python_requires='>=3.6',
)
