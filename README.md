# surrogate_article

Title: Geometric design of micro scale volumetric receiver using system-level inputs: An application of surrogate-based approach

Journal: Solar Energy

doi: [10.1016/j.renene.2022.12.105](https://doi.org/10.1016/j.solener.2023.111811)

link: [https://www.sciencedirect.com/science/article/pii/S0960148122019012](https://www.sciencedirect.com/science/article/pii/S0038092X2300436X)

This repo is looking as html dominant repo due to the n2 diagrams. It is actually pure python code and one n2 diagram.

This repository contains the test/validation data and and codes used in the article.

How to use this repo:

1. Creating the train and validation data:
User run 'DOE_recorder.py' file for data generation. Number samples must be set for design of experiements. This file outputs a .sql file. This is the output data of the DOE. You should rename (i.e. train.sql, validation.sql). Now, you got two different data for train and validation.

2. Creating .csv files from .sql files:
User run 'driver_postprocess.py' file for generating .csv files from each .sql files. User must relocate these .csv files in a folder for each .sql file.

3. Testing different surrogate models:
User run 'MDAOMetaModel.py' file for testing different surrogate models. For running the file train and validation data folders must be updated inside the Python file.
If you want to see pair plots 'data concat.py' file must be run for the data. This code combines all .csv files and creates single .csv file. After that you can uncomment pairplot codes (6 lines)

4. Running optimization with surrogate model:
User run 'MDAOMetaModel_Opt.py' file for surrogate-based optimization. User can alter the parameters inside the code. Do not miss to update train folder location in 'surr' class defined in the options as 'test_folder' parameter.

Python 3.9.13
Important pip dependencies:

Package                       Version
----------------------------- -----------
matplotlib                    3.5.2

numpy                         1.22.3

openmdao                      3.18.0

pandas                        1.4.2

pip                           21.2.4

scipy                         1.7.3

seaborn                       0.11.2


