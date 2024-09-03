from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name='defiant',
    description='DEFIANT (Data-driven Enhanced Frequentist Inference Analysis with Next-gen Techniques)',
    long_description=readme,
    long_description_content_type="text/markdown",
    version='0.1.0',
    author='Kyle A. Gersbach',
    author_email='gersbach.ka[at]gmail.com',
    url='https://github.com/GersbachKa/defiant/',
    packages=find_packages(),
    license="MIT license",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "healpy",
        "tqdm",
        "scipy",
        "enterprise-extensions",
        "la_forge",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    python_requires='>=3.9',
)