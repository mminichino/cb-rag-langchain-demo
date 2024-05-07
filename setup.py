from setuptools import setup, find_packages
import cbragdemo
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='cbragdemo',
    version=cbragdemo.__version__,
    packages=find_packages(exclude=['tests']),
    url='https://github.com/mminichino/cbragdemo',
    license='MIT License',
    author='Michael Minichino',
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'demo_prep = cbragdemo.demo_prep:main',
            'demo_run = cbragdemo.streamlit_exec:main',
            'demo_reset = cbragdemo.demo_reset:main'
        ]
    },
    install_requires=[
        "couchbase>=4.2.1",
        "streamlit>=1.32.2",
        "httpx>=0.27.0",
        "langchain>=0.1.13",
        "langchain-community>=0.0.29",
        "langchain-openai>=0.1.0",
        "tiktoken>=0.6.0",
        "pypdf>=4.1.0",
        "requests>=2.31.0",
        "cbcmgr>=2.2.40",
        "watchdog>=4.0.0"
    ],
    author_email='info@unix.us.com',
    description='Couchbase RAG Demo',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=["couchbase", "vector", "demo"],
    classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Developers",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Topic :: Software Development :: Libraries",
          "Topic :: Software Development :: Libraries :: Python Modules"],
)
