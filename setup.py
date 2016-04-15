from setuptools import setup, find_packages

setup(name='niftybold',
      version='0.1',
      description='BOLD-fMRI preprocessing and analysis tools',
      url='https://cmiclab.cs.ucl.ac.uk/mhutel/NiftyBOLD',
      author='Michael Hutel',
      author_email='michael.hutel.13@ucl.ac.uk',
      license='MIT',
      packages=find_packages(),
      install_requires=["numpy","scipy","sklearn","nibabel","nilearn"],
      zip_safe=False)
