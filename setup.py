from setuptools import setup

setup(name='protonets',
      version='0.0.1',
      author='Minzhao Liu',
      author_email='mliu6@uchicago.edu',
      license='MIT',
      packages=['protonets', 'protonets.utils', 'protonets.data', 'protonets.models'],
      install_requires=[
          'torch',
          'tqdm'
      ])
