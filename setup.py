from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="tuyabot_llm",
    version="0.1.0",
    description="TuyaBot-LLM is a Python-based chatbot that uses a large language model (LLM) and web data from a finance company to answer finance-related questions. It combines NLP with company-specific information to provide accurate, real-time responses to user queries.",
    package_dir={"":"tuyabot_llm"},
    author="John Mario Montoya Zapata",
    author_email="jmmontoyaz@unal.edu.co",
    long_description=long_description,
    url="URL or remote path of this project",
    packages=find_packages(where="tuyabot_llm"),
    python_requires=">=3.11.3",
    install_requires=install_requires, 
    extras_require={
        "dev": [
                "wheel==0.43.0",
                "notebook==7.2.1"
                ]
    },
)
