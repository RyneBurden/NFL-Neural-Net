# Staley - A Neural Network To Predict NFL Games


### **Overview**
Staley is a neural network used to predict NFL games based on datasets containing 29 features (14 for the home and away team, 1 to indicate whether the game is a divisional game) per game. Currently in its second iteration, the network is trained on the last 21 NFL seasons with a training batch size of 16. This batch size was chosen to hopefully optimize prediction accuracy on a by-week basis rather than a by-game basis, as predictions are made a week at a time (average of 15 to 16 games per week). Data is scaled to be zero-centered using the AbsMax function from the ScikitLearn Preprocessing python module, and this scaling is performed after seperarting the data into batches, so the input data should always be similar. Training and validation data are shuffled multiple times before each training epoch. I am not a data science researcher or professional, so all of my programming decisions are based on Machine Learning books/research and MIT online lectures from Lex Fridman.
  
  
### **How are predictions made?** 
5 models are trained on the same data before the season starts. A dataframe is made before each week containing a row for each game that will be played, and the dataframe is passed through each model. The results are compared, and the result that occurs most from the 5 models is considered the prediction. This method may change as I study version 2 a little more. It was implemented in version 1 due to the training batch size being 1; that small of a batch size along with the data not being scaled made it easier for the model to learn useless patterns in the training data.
  
  
### **2020-2021 Season Results** 
159-96-1 *or* 62.1% regular season prediction accuracry

### **Current training/validation loss and accuracy results** 


### **Dependencies** 
##### *Python Dependencies*
Matplotlib - used for graphing loss during training
Pytorch - used for creation and training of the network models
Beautiful Soup - Used for web scraping and stat gathering
Openpyxl - Used for spreadsheet manipulation and record-keeping
Scikit Learn - Used for data preprocessing
Numpy - Used for data processesing and manipulation

##### *R Dependencies*
nflfastR - used to gather and synthesize data from the 1999-2020 NFL seasons
Tidyverse - used to clean data after being gathered
Reticulate - used to interface R dataframes with the Python neural network


### **Twitter: @rjb_tech**
