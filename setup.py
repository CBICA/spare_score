from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='spare_scores',
      version='0.0.6',
      description='Compute characteristic brain signatures of your case population.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Gyujoon Hwang',
      author_email='ghwang1106@gmail.com',
      license='MIT',
      packages=['spare_scores'],
      install_requires=['numpy', 'pandas', 'sklearn']
      )