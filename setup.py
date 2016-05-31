from setuptools import setup, find_packages

setup(
    name='scsprox',

    version='0.1.0a1',

    description='Fast proximal operators from CVXPY problems with CySCS',
    long_description="Converts a CVXPY problem and a dict of CVXPY variables to a fast proximal operator object, which uses CySCS to provide fast evaluation, via one-time matrix stuffing, CySCS factorization caching, and automatic warm-starting of variables.",

    url='https://github.com/ajfriend/scsprox',

    author='AJ Friend',
    author_email='ajfriend@gmail.com',

    license='MIT',

    keywords='scs cvxpy convex optimization proximal operators ADMM',

    packages=['scsprox'],

    install_requires=['numpy', 'scipy', 'cvxpy', 'cyscs', 'pytest', 'psutil'],
)