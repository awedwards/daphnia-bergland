from setuptools import setup

setup(
        name="daphnia",
        version="0.1",
        py_modules = ['daphnia'],
        install_requires=['Click',
            'numpy==1.13.0',
            'scipy==0.19.1',
            'pandas==0.20.2',
            'opencv-python==3.2.0.7',
            ],
        entry_points='''
            [console_scripts]
            daphnia=daphnia:main
        ''',
)
