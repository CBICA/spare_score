from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='spare_scores',
      version='0.1.6',
      description='Compute characteristic brain signatures of your case population.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Gyujoon Hwang',
      author_email='ghwang1106@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data={'spare_scores':['mdl/*.pkl.gz','data/*.csv']},
      include_package_data=True,
      install_requires=['numpy', 'pandas', 'sklearn']
      )