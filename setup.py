import os

from setuptools import setup

requires = open("requirements.txt", "r").readlines() if os.path.exists("requirements.txt") else open("./azure_kinect_apiserver.egg-info/requires.txt", "r").readlines()
print("#-------------------    ", str(os.listdir("./")))
setup(
    name="azure-kinect-apiserver",
    version="0.0.1",
    author="davidliyutong",
    author_email="davidliyutong@sjtu.edu.cn",
    description="Azure Kinect APIServer",
    packages=[
        "azure_kinect_apiserver",
        "azure_kinect_apiserver.apiserver",
        "azure_kinect_apiserver.client",
        "azure_kinect_apiserver.cmd",
        "azure_kinect_apiserver.common",
        "azure_kinect_apiserver.decoder",
        "azure_kinect_apiserver.thirdparty",
    ],
    python_requires=">=3.7",
    install_requires=requires,
    test_requires=[
        "requests",
        "tqdm",
    ],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown"
)