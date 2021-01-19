import setuptools

setuptools.setup(
    name="CKG", # Replace with your own username
    version="1.0.0",
    author="Alberto Santos Delgado",
    author_email="alberto.santos@sund.ku.dk",
    description="A Python project that allows you to analyse proteomics and clinical data, and integrate and mine knowledge from multiple biomedical databases widely used nowadays.",
    url="https://github.com/MannLabs/CKG",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
