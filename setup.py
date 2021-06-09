from setuptools import setup, find_packages

setup(
    name='transition_sampling',
    version='0.0.1',
    packages=find_packages(include=['transition_sampling',
                                    'transition_sampling.*']),
    entry_points={
        'console_scripts': ['aimless_driver=transition_sampling.driver:main'],
    },
    url='',
    license='MIT',
    author='Isaiah Lemmon',
    author_email='lemmoi@uw.edu',
    description='MD engine agnostic Aimless Shooting algorithm'
)