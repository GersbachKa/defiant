from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name='defiant',
    description='DEFIANT (Data-driven Enhanced Frequentist Inference Analysis with Next-gen Techniques)',
    long_description=readme,
    long_description_content_type="text/markdown",
    version='0.2.0',
    author='Kyle A. Gersbach',
    author_email='gersbach.ka@gmail.com',
    url='https://github.com/GersbachKa/defiant/',
    packages=find_packages(),
    license="MIT license",
    install_requires=[
        "healpy>=1.15.0",
        "matplotlib>=3.0.0",
        "tqdm>=4.32.2",
        "la_forge>=1.1.0"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    python_requires='>=3.9',
)