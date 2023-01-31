#cleaning the data set

#libraries
import pandas as pd
import numpy as np

#read the data
data = pd.read_csv("austin_weather.csv")

#delete unnecessary data
data= data.drop(['Events','Date' , 'SeaLevelPressureHighInches','SeaLevelPressureLowInches'], axis=1)

data=data.replace('T',0.0)
data=data.replace('-',0.0)

data.to_csv('austin_final.csv')