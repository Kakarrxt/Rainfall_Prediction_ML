#Rainfall Prediction using Random Forest
#Location of data used for this => Albury

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
import seaborn as sns

full_data = pd.read_csv('weatherAUS.csv')
full_data.head()


full_data['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
full_data['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

no = full_data[full_data.RainTomorrow == 0]
yes = full_data[full_data.RainTomorrow == 1]
yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
oversampled = pd.concat([no, yes_oversampled])



total = oversampled.isnull().sum().sort_values(ascending=False)
percent = (oversampled.isnull().sum()/oversampled.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing.head(4)


 #Imputing missing data using MICE (Multiple Imputation by Chained Equations ) 

oversampled.select_dtypes(include=['object']).columns

# Impute categorical var with Mode
oversampled['Date'] = oversampled['Date'].fillna(oversampled['Date'].mode()[0])
oversampled['Location'] = oversampled['Location'].fillna(oversampled['Location'].mode()[0])
oversampled['WindGustDir'] = oversampled['WindGustDir'].fillna(oversampled['WindGustDir'].mode()[0])
oversampled['WindDir9am'] = oversampled['WindDir9am'].fillna(oversampled['WindDir9am'].mode()[0])
oversampled['WindDir3pm'] = oversampled['WindDir3pm'].fillna(oversampled['WindDir3pm'].mode()[0])

# Convert categorical features to continuous features with Label Encoding
from sklearn.preprocessing import LabelEncoder
lencoders = {}
for col in oversampled.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    oversampled[col] = lencoders[col].fit_transform(oversampled[col])

import warnings
warnings.filterwarnings("ignore")

# Multiple Imputation by Chained Equations
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
MiceImputed = oversampled.copy(deep=True) 
mice_imputer = IterativeImputer()
MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversampled)

# Detecting outliers with IQR
Q1 = MiceImputed.quantile(0.25)
Q3 = MiceImputed.quantile(0.75)
IQR = Q3 - Q1

# Removing outliers from the dataset
MiceImputed = MiceImputed[~((MiceImputed < (Q1 - 1.5 * IQR)) |(MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]
MiceImputed.shape




# Removing outliers from the dataset
MiceImputed = MiceImputed[~((MiceImputed < (Q1 - 1.5 * IQR)) |(MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]
MiceImputed.shape


# Standardizing data
from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(MiceImputed)
modified_data = pd.DataFrame(r_scaler.transform(MiceImputed), index=MiceImputed.index, columns=MiceImputed.columns)

# Feature Importance using Filter Method (Chi-Square)
from sklearn.feature_selection import SelectKBest, chi2
X = modified_data.loc[:,modified_data.columns!='RainTomorrow']
y = modified_data[['RainTomorrow']]
selector = SelectKBest(chi2, k=10)
selector.fit(X, y)
X_new = selector.transform(X)





#Training Rainfall Prediction Model with Random Forest


features = MiceImputed[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']]
target = MiceImputed['RainTomorrow']

# Split into test and train (85% for training 15% for testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.15, random_state=12345)

# Normalize Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, roc_curve, classification_report
def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    
    return model, accuracy, roc_auc, coh_kap, time_taken




#Training using Random Forest
from sklearn.ensemble import RandomForestClassifier

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

model_rf = RandomForestClassifier(**params_rf)
model_rf, accuracy_rf, roc_auc_rf, coh_kap_rf, tt_rf = run_model(model_rf, X_train, y_train, X_test, y_test)

#Accuracy= 0.9555649597968684
#95.5%



 #Testing Random Data using the existing Data
data = {
        'MinTemp': [7.7], 
        'MaxTemp': [26.7], 
        'Rainfall':[0],
        'Evaporation': [5.85812957879808], 
        'Sunshine': [11.9605549245276], 
        'WindGustDir': [13],
         'WindGustSpeed': [35], 
        'WindDir9am': [10], 
        'WindDir3pm': [13], 
        'WindSpeed9am': [6], 
        'WindSpeed3pm': [17], 
        'Humidity9am': [48], 
        'Humidity3pm': [19], 
        'Pressure9am': [1010.8], 
        'Pressure3pm': [1008.6], 
        'Cloud9am': [1.55340991681667], 
        'Cloud3pm': [2.26956647477166], 
        'Temp9am': [16.3], 
        'Temp3pm': [25.5],
        'RainToday':[0]}
input = pd.DataFrame(data)


# Make the prediction
prediction = model_rf.predict_proba(input)[0,1]

print("prediction= {}".format(prediction))

#gives probability => 0.29894736842105263 

if prediction<(0.25):
    print("No rain is expected tomorrow")
else:
    print("Rain is expected tomorrow") 




#Taking User Input

min_temp = float(input("Enter the minimum temperature: "))
max_temp = float(input("Enter the maximum temperature: "))
Rainfall = float(input("Enter the Rainfall: "))
evaporation = float(input("Enter the evaporation: "))
sunshine = float(input("Enter the sunshine: "))
wind_gust_dir = input("Enter the wind gust direction: ")
wind_gust_speed = float(input("Enter the wind gust speed: "))
wind_dir_9am = input("Enter the wind direction at 9am: ")
wind_dir_3pm = input("Enter the wind direction at 3pm: ")
wind_speed_9am = float(input("Enter the wind speed at 9am: "))
wind_speed_3pm = float(input("Enter the wind speed at 3pm: "))
humidity_9am = float(input("Enter the humidity at 9am: "))
humidity_3pm = float(input("Enter the humidity at 3pm: "))
pressure_9am = float(input("Enter the pressure at 9am: "))
pressure_3pm = float(input("Enter the pressure at 3pm: "))
cloud_9am = float(input("Enter the cloudiness at 9am: "))
cloud_3pm = float(input("Enter the cloudiness at 3pm: "))
temp_9am = float(input("Enter the temperature at 9am: "))
temp_3pm = float(input("Enter the temperature at 3pm: "))
RainToday = float(input("Enter the Rain today: "))

# Convert the user input into a dataframe
data = {'MinTemp': [min_temp], 
        'MaxTemp': [max_temp], 
        'Rainfall': [Rainfall],
        'Evaporation': [evaporation], 
        'Sunshine': [sunshine], 
        'WindGustDir': [wind_gust_dir],
        'WindGustSpeed': [wind_gust_speed], 
        'WindDir9am': [wind_dir_9am], 
        'WindDir3pm': [wind_dir_3pm], 
        'WindSpeed9am': [wind_speed_9am], 
        'WindSpeed3pm': [wind_speed_3pm], 
        'Humidity9am': [humidity_9am], 
        'Humidity3pm': [humidity_3pm], 
        'Pressure9am': [pressure_9am], 
        'Pressure3pm': [pressure_3pm], 
        'Cloud9am': [cloud_9am], 
        'Cloud3pm': [cloud_3pm], 
        'Temp9am': [temp_9am], 
        'Temp3pm': [temp_3pm],
        'RainToday': [RainToday]}

user_input = pd.DataFrame(data)


# Make the prediction
prediction = model_rf.predict_proba(user_input)[0,1]

print("prediction= {}".format(prediction))

#gives probability => 0.29894736842105263 

if prediction<(0.25):
    print("No rain is expected tomorrow")
else:
    print("Rain is expected tomorrow") 