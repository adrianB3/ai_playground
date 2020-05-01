import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ai_playground",
    version="0.0.2",
    author="Adrian Bălănescu",
    author_email="balanescuadrian71@gmail.com",
    description="AI playground reinforcement learning framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://dev.azure.com/codeworks5/ai_playground/_git/ai_playground",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': ['aip = ai_playground.cli:start']
    }
)
