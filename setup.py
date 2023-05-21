import os

from setuptools import setup, find_packages

requires = open("requirements.txt", "r").readlines() if os.path.exists("requirements.txt") else open("./azure_kinect_apiserver.egg-info/requires.txt", "r").readlines()
print("#-------------------    ", str(os.listdir("./")))
setup(
    name="azure-kinect-apiserver",
    version="0.2.2",
    author="davidliyutong",
    author_email="davidliyutong@sjtu.edu.cn",
    description="Azure Kinect APIServer",
    packages=find_packages() + ["azure_kinect_apiserver.thirdparty.pyKinectAzure." + pkg for pkg in find_packages('azure_kinect_apiserver/thirdparty/pyKinectAzure')],
    python_requires=">=3.7",
    install_requires=requires,
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown"
)
