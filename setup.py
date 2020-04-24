from setuptools import setup

setup(
    name="nanolyse",
    version="v0.0.1",
    author="Florian L.R. Lucas",
    packages=["nanolyse"],
    install_requires=[
        'numpy >= 1.18.2',
        'scipy >= 1.4.1'
    ],
    extra_requires={
        "Axon": 'neo >= 0.8.0'
    },
    python_requires='>=3.7'
)