# Rainfall_Prediction_ML

<h3><b>TL;DR</b></h3>
Using machine learning(random forest algorithm) created a Rainfall Prediction Project that takes user input to predict if theres going to be rainfall tomorrow or not.

In training after using the Random Forest the program was accurute upto 95%


<h3><b>Methodology:</b></h3>
The data used for this project was collected from the Australian Meteorological Department for the past 10 years. 
It contained information about the monthly rainfall in a specific region. The data was preprocessed to remove any missing values and outliers.

The random forest algorithm was then trained on the preprocessed data using the scikit-learn library in Python. 
The algorithm was trained using 85% of the data, and the remaining 15% was used for testing.

<h3><b>User Input:</b></h3>
The user can input the weather conditions to predict if there's going to be rainfall tomorrow or not.


<h3><b>Testing:</b></h3>
The Processed data was tested on 7 different algorithms which includes 
<ul>
<li>Logistic Regression</li>
<li>Decision Tree</li>
<li>Neural Network</li>
<li>Random Forest</li>
<li>LightGBM</li>
<li>CatBoost</li>
<li>XGBoost</li>
</ul>

With Highest accuracy being of XGBoost and fastest being Decision Tress<br>
The lowest accuracy was of Logistic Regression and slowest was Nueral Network
<br>
<br>
We used Random forest in the end as it had above 95% accuracy and being one of the quickest algorithms 

<h3><b>Results:</b></h3>
The accuracy of the model was found to be around <b>95%</b>. This means that the model was able to correctly predict the rainfall in 95% of the test cases. 
The results were visualized using a scatter plot, where the actual rainfall was plotted against the predicted rainfall.

<h3><b>Conclusion:</b></h3>
The random forest algorithm was able to provide a good accuracy in predicting the rainfall in a specific region. 
This project shows the potential of machine learning algorithms in weather forecasting. 
Further improvements can be made by incorporating additional weather parameters and increasing the size of the data set used for training.
