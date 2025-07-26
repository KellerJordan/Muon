from setuptools import setup, find_packages

setup(
    name='muon-optimizer',
    version='0.1.0',
    author='Keller Jordan',
    author_email='kjordan4077@gmail.com',
    description='Muon opimizer',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KellerJordan/Muon',
    #packages=find_packages(),
    py_modules=['muon'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
