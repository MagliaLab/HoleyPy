from setuptools import setup

setup(
    name="holeypy",
    version="v0.0.1",
    author=["Florian L.R. Lucas", "Matthijs J. Tadema"],
    packages=["holeypy"],
    install_requires=[
        'numpy >= 1.18.2',
        'scipy >= 1.4.1',
        'h5py >= 2.10.0',
        'neo >= 0.8.0',
        'pandas',
        'openpyxl',
        'pyabf'
    ],
    extra_requires={
        "Axon": 'neo >= 0.8.0'
    },
    python_requires='>=3.7',
    package_data={
        '': [
            "tests/data/Blank.csv",
            "tests/data/ProteinDigest.csv",
            "tests/data/Blank.abf",
            "tests/data/ProteinDigest.abf"
        ]
    }

)
