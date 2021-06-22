<h1 align="center"> Staley - A Neural Network To Predict NFL Games </h1>

<h3> **Overview** </h3>
<p>Staley is a neural network used to predict NFL games based on datasets containing 29 features (14 for the home and away team, 1 to indicate whether the game is a divisional game) per game. Currently in its second iteration, the network is trained on the last 21 NFL seasons with a training batch size of 16. This batch size was chosen to hopefully optimize prediction accuracy on a by-week basis rather than a by-game basis, as predictions are made a week at a time (average of 15 to 16 games per week). Data is scaled to be zero-centered using the AbsMax function from the ScikitLearn Preprocessing python module, and this scaling is performed after seperarting the data into batches, so the input data should always be similar. Training and validation data are shuffled multiple times before each training epoch. I am not a data science researcher or professional, so all of my programming decisions are based on Machine Learning books/research and MIT online lectures from Lex Fridman.<p>
  
<h3> **How are predictions made?** </h3>
<p>5 models are trained on the same data before the season starts. A dataframe is made before each week containing a row for each game that will be played, and the dataframe is passed through each model. The results are compared, and the result that occurs most from the 5 models is considered the prediction. This method may change as I study version 2 a little more. It was implemented in version 1 due to the training batch size being 1; that small of a batch size along with the data not being scaled made it easier for the model to learn useless patterns in the training data.
  
<h3> **2020-2021 Season Results** </h3>
<p>159-96-1 *or* 62.1% regular season prediction accuracry</p>

<h3> **Current training/validation loss and accuracy results** </h3>

<h3> **Dependencies** </h3>
<p>*Python Dependencies*</p>
<p>Matplotlib - used for graphing loss during training</p>
<p>Pytorch - used for creation and training of the network models</p>
<p>Beautiful Soup - Used for web scraping and stat gathering</p>
<p>Openpyxl - Used for spreadsheet manipulation and record-keeping</p>
<p>Scikit Learn - Used for data preprocessing</p>
<p>Numpy - Used for data processesing and manipulation</p>

<p>*R Dependencies*</p>
<p>nflfastR - used to gather and synthesize data from the 1999-2020 NFL seasons</p>
<p>Tidyverse - used to clean data after being gathered</p>
<p>Reticulate - used to interface R dataframes with the Python neural network</p>

<h3> **Twitter: @rjb_tech** </h3>
