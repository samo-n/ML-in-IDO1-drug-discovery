# IDO1 drug discover using ML
In this repository I am sharing the work I did as part of my master's thesis "Discovery of new IDO1 inhibitors with machine learning"

## Why?

In order to develop new inhibitors of the enzyme indoleamine 2,3-dioxygenase 1 (IDO1), which represents a promising target for cancer immunotherapy, we will focus on the use of machine learning in our master's thesis.
From various publicly available databases, such as ChemBl and BindingDB, we will obtain data on molecules already tested for IDO1, and then we will analyze them in the Python programming language and identify properties that distinguish active from inactive molecules. In the next part, we will make a binary machine learning classification model that will predict whether a certain 3D structure is active or inactive. In order for the data of molecular 3D structures to be suitable for training a machine learning model,
we will first discretize them in several ways into a bit record called "fingerprint". This will be followed by processing with several steps of preprocessing and finding the best machine learning classification model. Finally, we will apply the best model to 3D structures that have not yet been tested on the IDO1 enzyme, obtain the best potential hits and analyze the properties of potential new IDO1 inhibitors.
