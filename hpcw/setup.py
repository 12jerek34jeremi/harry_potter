from setuptools import setup

setup(
   name='hpcw',
   version='0.1.0',
   author='Jedrzej Chmiel',
   author_email='jedrzej.chmiel.ml@gmail.com',
   install_requires=[
      "torch",
      "nltk == 3.7"
   ],
   packages=['hpcw']
)