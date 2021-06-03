import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='DAFI',
    version='1.0.1',
    packages=setuptools.find_packages(),
    scripts=['bin/dafi'],
    license='Apache 2.0',
    author="Virginia Tech",
    description="Ensemble based data-assimilation and field inversion.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/xiaoh/DAFI',
    download_url="https://github.com/xiaoh/DAFI/archive/refs/tags/1.0.0.tar.gz",
    keywords=["Data Assimilation", "Field Inversion"],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pyyaml',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
    ],
)
