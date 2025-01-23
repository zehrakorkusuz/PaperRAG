from setuptools import setup, find_packages

setup(
    name='paperrag',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'nest_asyncio',
        'numpy',
        'rank_bm25',
        'torch',
        'clip',
        'Pillow',
        'matplotlib==3.10'
    ],
    author='Zehra Korkusuz, Kuan-Lin Huang',
    author_email='wzehrakorkusuz@gmail.com',
    description='CPU efficient RAG model for scientific documents',
    url='https://github.com/zehrakorkusuz/PaperRAG',  #
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: MIT License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)