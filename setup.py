from setuptools import setup
# from distutils.core import setup

setup(name='lemonpy',
      packages=['lemonpy'],
      version='0.1.0',
      description='Template for new Python packages',
      author='Theodore Tsitsimis',
      author_email='th.tsitsimis@gmail.com',
      url='https://github.com/tsitsimis/lemonpy',
      download_url='https://github.com/tsitsimis/lemonpy/archive/0.0.2.tar.gz',
      keywords=['skeleton', 'package'],
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 3.4'
      ],
      install_requires=[
          'numpy',
      ],
      zip_safe=False
      )
