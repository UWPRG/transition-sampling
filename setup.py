from setuptools import setup, find_packages

setup(
    name='transition_sampling',
    version='0.0.1',
    packages=find_packages(include=['transition_sampling',
                                    'transition_sampling.*']),
    url='',
    license='MIT',
    author='Isaiah Lemmon',
    author_email='lemmoi@uw.edu',
    description='MD engine agnostic Aimless Shooting algorithm'
)