from setuptools import setup, find_packages

setup(name='pbrl-ingredients',
      version='0.1',
      description='Provides common sacred ingredients for preference-based reinforcement learning experiments.',
      url='https://github.com/mschweizer/pbrl-ingredients',
      author='Marvin Schweizer',
      author_email='schweizer@kit.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'sacred==0.8.4',
          'imitation==0.3.2',
      ],
      include_package_data=True,
      python_requires='>=3.7',
      )
