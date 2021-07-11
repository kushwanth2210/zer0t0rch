from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='zerotorch',
    version='1.0.0',
    description='a lightweight pytorch wrapper for research and development purpose',
    long_description=open('README.md').read() + '\n\n' +
    open('CHANGELOG.txt').read(),
    url='',
    author='zEro',
    author_email='',
    license='MIT',
    classifiers=classifiers,
    keywords='machinelearning',
    packages=find_packages(),
    install_requires=['tqdm', 'numpy', 'livelossplot']
)
