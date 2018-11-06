from setuptools import setup

setup(name='clustergb',
      version='1.0.0',
      description='Vacuum-cluster LAMMPS calculations of grain boundaries (GBs)',
      url='https://github.com/liamhuber/clustergb',
      author='Liam Huber',
      author_email='huber@mpie.de',
      license='MIT',
      packages=['clustergb'],
      install_requires=['numpy>=1.11.2', 'scipy>=0.17.1', 'yaml'],
      zip_safe=False)
