from setuptools import find_packages
from setuptools import setup

setup(name='slip',
      version='0.0.0.1',
      description='Functions that work with strike-slip_model.py strike-slip landscape evolution model',
      author='Nadine Reitman',
      author_email='nadine.reitman@colorado.edu',
      url='',
      #packages=['slip'],
      packages=find_packages(where="src"),
      package_dir={'': 'src'},
      install_requires=['numpy','matplotlib'],
      #entry_points={'console_scripts': ['calcOFD = slip.slip:calcOFD']},
      #url='ttps://github.com/USER/REPO',
      zip_safe=False
     )
