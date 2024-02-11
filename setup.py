from setuptools import find_packages, setup

setup(
    name="mcq_generator",
    version='0.0.1',
    author='Hari',
    author_email='harikrishnan1998@gmail.com',
    install_requires=['openai', 'langchain', 'streamlit', 'python-dotenv', 'PyPDF2'],
    packages=find_packages()
)
