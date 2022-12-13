from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='spare_scores',
      version='0.0.1',
      description='Train SPARE classification or regression models and calculate SPARE scores.',
      long_description=readme(),
      author='Gyujoon Hwang',
      author_email='ghwang1106@gmail.com',
      license='MIT',
      packages=['spare_scores'],
      install_requires=['numpy', 'pandas', 'sklearn'],
      zip_safe=False)