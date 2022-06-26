from setuptools import find_packages, setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Reinforcement learning methodologies for optomechanical system.',
    author='GOL - PUC-RJ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/oscarkremer/deep-tweezer',
)