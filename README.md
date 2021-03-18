# QNN4O: a quantum convolutional neural network for satellite data classification

Authors:
* Daniela A. Zaidenberg (MIT)
* Alessandro Sebastianelli (Univeristy of Sannio/ESA OSIP)
* Dario Spiller (ASI/ESA)
* Bertrand Le Saux (ESA)
* Silvia L. Ullo (University of Sannio)

### Installation

This module has been implemented in [Python 3.6.8](https://www.python.org/downloads/release/python-368/).

After the installation of Python and [pip](https://pypi.org/project/pip/), you can clone this repository in your working directory.

Then you can create a virtual environment:
1. open your favourite terminal and navigate into your working directory
2. install the *virtualenv* command: `pip install virtualenv`
3. create a virtualenv: `virtualenv qnn4eo -p python3.6`
4. activate the virtualenv: 
    - Linux/MacOS: `source qnn4eo\bin\activate`
    - Windows: `qnn4eo/Scripts/activate`
5. install requirements: `pip install -r requirements.txt`
6. launch Jupyter Lab and open EuroSAT_Classification.ipynb: `jupyter lab`



## Some results

![](imgs/qnnVScnn.png)

