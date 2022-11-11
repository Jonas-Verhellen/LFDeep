from setuptools import setup, find_packages

setup(
    name='LFDeep',
    version='0.0.1',
    description='Multi-Expert Deep learning of Votlage Traces and Spike Times In A Cortical Neuron',
    author='',
    author_email='',
    url='https://github.com/Jonas-Verhellen/LFDeep',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

