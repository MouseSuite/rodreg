# Rodent Brain Registration Project

This project is focused on the registration of rodent brain images using various techniques and tools.

## Table of Contents

- [Rodent Brain Registration Project](#rodent-brain-registration-project)
  - [Table of Contents](#table-of-contents)
  - [Quick Summary](#quick-summary)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
    - [Dependencies](#dependencies)
    - [Configuration](#configuration)
    - [Running Tests](#running-tests)
    - [Deployment Instructions](#deployment-instructions)
  - [Usage](#usage)
  - [Contribution Guidelines](#contribution-guidelines)
  - [Contact](#contact)

## Quick Summary

This repository contains scripts and notebooks for performing image registration on rodent brain images. The project utilizes tools such as SimpleITK, Nilearn, and MONAI for image processing and registration tasks.

## Project Structure


## Setup

### Dependencies

Install the required dependencies using the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

### Configuration
Ensure that the paths to the input images and atlases are correctly set in the scripts and notebooks.


### Usage
Use the Jupyter notebooks for interactive exploration and registration tasks.
Run the scripts for batch processing and automated registration.
Example command to run a script:

```
python run_rodreg.py -i input_image.nii.gz -r reference_image_prefix --o output_image.nii.gz --l output_label.nii.gz
```

### Contribution Guidelines
#### Writing Tests
- Write tests for new features and bug fixes.
- Ensure all tests pass before submitting a pull request.
#### Code Review
- Follow the project's coding standards.
- Ensure code is well-documented and readable.
### Other Guidelines
- Open an issue to discuss proposed changes before submitting a pull request.
- Provide detailed descriptions of changes in pull requests.


