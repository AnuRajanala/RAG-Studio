from setuptools import setup, find_packages

with open('requirements.in') as f:
    requirements = f.read().splitlines()

setup(
    name='rag-studio',
    version='1.0',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
       
    },
    description='RAG Studio is a Streamlit app for knowing the inner workings of RAG',
    #long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    #url='https://github.com/your-username/rag-studio'
)