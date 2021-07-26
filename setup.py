from setuptools import setup, find_packages

setup(
    name='scsprox',

    version='0.2.0',

    description='Fast proximal operators from CVXPY problems with CySCS',
    long_description="Converts a CVXPY problem and a dict of CVXPY variables to a fast proximal operator object, which uses CySCS to provide fast evaluation, via one-time matrix stuffing, CySCS factorization caching, and automatic warm-starting of variables.",

    url='https://github.com/ajfriend/scsprox',

    author='AJ Friend',
    author_email='ajfriend@gmail.com',

    license='MIT',

    keywords='scs cvxpy convex optimization proximal operators ADMM',

    packages=['scsprox'],
    package_data={'scsprox': ['test/*.py']},
    zip_safe=False,  # apparently, this is needed to include the test dir

    install_requires=['numpy', 'scipy', 'cvxpy >= 1.1', 'cyscs', 'pytest', 'psutil'],
)