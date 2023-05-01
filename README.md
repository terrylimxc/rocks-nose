# DSA4262: rocks-nose
[![GitHub Super-Linter](https://github.com/terrylimxc/rocks-nose/actions/workflows/linter.yml/badge.svg)](https://github.com/marketplace/actions/super-linter) 
![GitHub pull requests](https://img.shields.io/github/issues-pr/terrylimxc/rocks-nose)
![GitHub repo size](https://img.shields.io/github/repo-size/terrylimxc/rocks-nose)
![GitHub last commit](https://img.shields.io/github/last-commit/terrylimxc/rocks-nose)

Task 1: Develop a machine learning method to identify RNA modifications from direct RNA-Seq data  
Task 2: Prediction of m6A sites in all SG-NEx direct RNA-Seq samples  

## Table of Contents
* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Contributing](#contributing)
* [Team](#team)
* [Acknowledgements](#acknowledgements)
* [License](#license)

## Introduction
Post-transcriptional modifications of RNA play a huge role in various biological processes  and disease progression, attracting accumulating attention in bioscience research. Among the many different RNA modifications, N6-methyladenosine (m6A) is the most commonly seen mRNA modification. It was reported that on average, 1000 nucleotides are found to contain 1–2 m6A residues. m6A is the methylation that occurs in the N6-position of adenosine, which is the most prevalent internal modification on eukaryotic mRNA. Strong evidence suggests that m6A modulates gene expression and influences the corresponding cell processes and physiological function. m6A affects almost all processes of RNA metabolism, ranging from mRNA translation, degradation to splicing and folding. Emerging evidence has revealed that this m6A modification is closely associated with the activation and inhibition of tumour pathways, and it is significantly linked to the prognosis of cancer patients.  With studies highlighting how restoring the balance of m6A modifications could be a novel anti-cancer strategy, there is a pressing need to accurately detect possible m6A modifications.

## Getting Started
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Configuration
Firstly, launch an AWS instance. Refer to this [tutorial](Research%20Gateway%20Tutorial.docx)  

Navigate to your home directory:  
```
cd ~
```

Install Python
```
sudo apt-get install python3-pip  
```

Git clone this repository
```
git clone https://github.com/terrylimxc/rocks-nose.git
```

Then change into the `rocks-nose` folder
```
cd rocks-nose
```

Create a virtual environment by executing the following commands:
```
# Install Virtual Environment
sudo pip3 install pipenv

# Check pipenv has been installed successfully  
pipenv --version  

# Initialise a Virtual Environment of your choice
pipenv shell

# Install all dependencies
pipenv install --dev
```

To exit the virtual environment
```
exit
```

To delete the virtual environment
```
pipenv --rm
```

## Usage
The time taken for each command takes around 2 minutes to execute. 
Sample training data and testing data have been provided in the `data` folder.
All results can be retrieved in the `results` folder.

### Train
```
python prepare.py -l data.info -d sample_training.json -o model
```
#### Options
```
--help  -h              Show help message and exit

--label -l              Specify file containing labels for training data

--data  -d              Specify training data

--model -m              Choice of model to use for training (SmoteTomek or BalancedRF)
                        Default: SmoteTomek

--output -o             Specify name of saved model
```

### Predict
```
python predict.py -d sample_testing.json -m model 
```
#### Options
```
--help  -h              Show help message and exit

--data  -d              Specify testing data

--model -m              Specify saved model to be used for testing
```


## Contributing
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/terrylimxc/rocks-nose) ![GitHub closed issues](https://img.shields.io/github/issues-closed/terrylimxc/rocks-nose) ![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/terrylimxc/rocks-nose)  
All source code is hosted on [GitHub](https://github.com/terrylimxc/rocks-nose). Contributions are welcome.

Contributions to the project are made using the "Fork & Pull" model. The typical steps would be:
1. Create an account on [GitHub](https://github.com)
2. Fork [`repo`](https://github.com/terrylimxc/rocks-nose)
3. Make a local clone: `git clone https://github.com/your_account/rocks-nose.git`
4. Make changes on the local copy
5. Test and commit changes `git commit -a -m "my message"`
6. `push` to your GitHub account: `git push origin`
7. Create a Pull Request (PR) from your GitHub fork
(go to your fork's web page and click on "Pull Request."
You can then add a message to describe your proposal.)
8. Wait for lint tests to pass. In case of failures, go back to (5)
9. If all is good, a maintainer will review your PR :)

### Style Guide
Official Style Guide: [Pep 8](http://www.python.org/dev/peps/pep-0008/)

## Team
* [Chan Wen Yong](https://github.com/wenyong13)
* [Lim  Xi Chen Terry](https://github.com/terrylimxc)
* [Teo Yoke Sheng Clifton](https://github.com/cliftontys)
* [Torrin G Panicker](https://github.com/Torrinp)
* [Zou Run Zhong](https://github.com/Zourunzhong)
  
## Acknowledgements

### Teaching Team
* [Prof Jonathan Göke](https://github.com/jonathangoeke)
* [TA Christopher Hendra](https://github.com/chrishendra93)
* [TA Yuk Kei](https://github.com/yuukiiwa)

### Resources
[SG-NEx](https://github.com/GoekeLab/sg-nex-data), [AWS](https://github.com/aws), [argparse](https://docs.python.org/3/library/argparse.html), 
[orjson](https://github.com/ijl/orjson), [xgboost](https://xgboost.readthedocs.io/en/stable/), [imbalanced-learn](https://imbalanced-learn.org/)

## License
 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
The content of this project itself is licensed under the [MIT license](https://github.com/terrylimxc/rocks-nose/blob/terry/LICENSE).
