import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score


# Reading CSV data from a file
heart_attack = pd.read_csv("heart_attack_prediction_dataset.csv")
print(heart_attack.info())
print(heart_attack.head())
print(heart_attack.isnull().sum().sort_values())

heart_attack["Sex"]=heart_attack["Sex"].astype("category")
print(heart_attack["Continent"].value_counts())
heart_attack["Continent"]=heart_attack["Continent"].astype("category")
print(heart_attack["Diet"].value_counts())
heart_attack["Diet"]=heart_attack["Diet"].astype("category")
print(heart_attack["Family History"].value_counts())


print(heart_attack.dtypes)

# plt.figure(figsize=(10,6))
# palette = sns.color_palette("deep")
# sns.countplot(data=heart_attack, x="Continent", hue="Heart Attack Risk", palette=palette)
# plt.show()

# plt.figure(figsize = (22,10))
# sns.heatmap(heart_attack[['Age','Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
#       'Cholesterol','Alcohol Consumption', 'Exercise Hours Per Week',
#      'Previous Heart Problems', 'Medication Use', 'Stress Level',
#         'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 'Sleep Hours Per Day','Heart Attack Risk']].corr(), cmap="YlGnBu",
#             annot=True)
# plt.show()

# #Continents vs.Cholestrol Level range
# plt.figure(figsize=(8,5))
# sns.set_theme(style="ticks", palette="pastel")
# sns.boxplot(x="Continent", y="Cholesterol", hue="Sex", palette=["m", "g"], data=heart_attack)
# plt.show()

# #Continents vs Exercise Hours Per Week range
# plt.figure(figsize=(8,5))
# sns.set_theme(style="ticks", palette="pastel")
# sns.boxplot(x="Continent", y="Exercise Hours Per Week", hue="Alcohol Consumption",palette=["b", "r"], data=heart_attack)
# plt.show()


continent_by_bmi=heart_attack.groupby("Continent")[["BMI"]].mean().sort_values(by="BMI", ascending=True).round(2)
country_by_sleep_activity = heart_attack.groupby("Country")[["Sleep Hours Per Day", "Physical Activity Days Per Week"]].mean().sort_values(by="Sleep Hours Per Day", ascending=False).round(2)
print(country_by_sleep_activity)
print(continent_by_bmi)

continent_by_stress = heart_attack.groupby("Continent")[["Stress Level"]].mean().sort_values(by="Stress Level", ascending=False)
print(continent_by_stress)

# plt.figure(figsize=(9,6))
# fig = px.scatter(heart_attack, title="Cholesterol vs. Triglycerides", x=heart_attack["Cholesterol"], y=heart_attack["Triglycerides"], hover_data=["Smoking"], color="Sex", size="Sleep Hours Per Day")
# fig.show()

# plt.figure(figsize=(9,6))
# fig = px.pie(heart_attack, names=heart_attack["Continent"], values="BMI", 
#              color_discrete_sequence=px.colors.qualitative.Set2, title="Continents by BMI Values")
# fig.show()

# plt.figure(figsize=(9,6))
# fig = px.pie(heart_attack, names=heart_attack["Country"], values="Sedentary Hours Per Day", 
#              color_discrete_sequence=px.colors.qualitative.Set3, title="Country by Sedentary Hours Per Day Values")
# fig.show()

# plt.figure(figsize=(9,6))
# ax = sns.scatterplot(data=heart_attack, x="Stress Level", y="Sedentary Hours Per Day", hue="Smoking")
# plt.show()


# plt.figure(figsize=(9,6))
# ax = sns.scatterplot(data=heart_attack, x="Stress Level", y="Exercise Hours Per Week", hue="Smoking")
# plt.show()


# plt.figure(figsize=(9,6))
# fig = px.scatter(heart_attack, x=heart_attack["Cholesterol"], y=heart_attack["Triglycerides"], hover_data="Stress Level", size="Age",
#              color="Stress Level", hover_name="Sex", custom_data="Income", opacity=0.8)
# fig.show()


print(heart_attack.groupby(["Continent", "Sex"])[["Heart Rate"]].mean().round(2))

#Correlation matrix
heart_attack["Sex"] = heart_attack["Sex"].str.replace("Female", "1").str.replace("Male","0").astype(int)

heart_attack_corr = heart_attack[["Sleep Hours Per Day", "Triglycerides", "BMI", "Income", "Age", "Medication Use", "Alcohol Consumption",
           "Obesity", "Smoking", "Diabetes","Exercise Hours Per Week", "Previous Heart Problems", "Sex"]]
corr_matrix = heart_attack_corr.corr().round(2)
print(corr_matrix)
sns.heatmap(corr_matrix,annot=True, cbar=True, fmt=".2f")
plt.show()


#Splitting Between Diastolic and Systolic Blood Pressure"""
heart_attack['BP_Systolic'] = heart_attack['Blood Pressure'].apply(lambda x: x.split('/')[0])
heart_attack['BP_Diastolic'] = heart_attack['Blood Pressure'].apply(lambda x: x.split('/')[1])
heart_attack.drop("Blood Pressure", axis=1, inplace=True)
print(heart_attack.info())

#Model Building 

X = heart_attack[['Age', 'Cholesterol', 'Heart Rate',
        'Diabetes', 'Family History', 'Smoking', 'Obesity',
        'Alcohol Consumption', 'Exercise Hours Per Week',
        'Previous Heart Problems', 'Medication Use', 'Stress Level',
        'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
        'Physical Activity Days Per Week', 'Sleep Hours Per Day',
         'BP_Systolic', 'BP_Diastolic','Sex']]
        
y=heart_attack['Heart Attack Risk'].values

print(X.head())
print(y[1:10])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Applying and Checking Accuracy and Precision of different Classification Models
models = {"Logistic_Regression":LogisticRegression(),
          "Random_Forest":RandomForestClassifier(),
          "SVM":SVC(kernel="rbf"),
          "KNN": KNeighborsClassifier(n_neighbors=3)}

for i in models :
    y_pred = models[i]
    y_pred.fit(X_train, y_train)
    y_pred_predict = y_pred.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_predict)
    precision = precision_score(y_test, y_pred_predict, zero_division=1)
    print('Accuracy of '+i+': ',accuracy)
    print('Precision of '+i+': ',precision) #Random Forest Classification Model suits this case the most

#Cross Validation on Random Forest Algorithms
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
accuracies = cross_val_score(estimator=rf, X=X_train, y=y_train, cv=10)
print(accuracies)
print(accuracies.mean())

#Hyperparameter Tuning of Random Forest Model
# #"""Using GridSearchCV function"""
# grid = [{'n_estimators':[100,300], 'max_depth':[5,10],
#         'min_samples_split':[2,5] , 'min_samples_leaf':[2,4]}]

# grid_search = GridSearchCV(estimator=rf, param_grid=grid, scoring="accuracy", cv=8, n_jobs = -1,verbose = 2)
# grid_search = grid_search.fit(X_train, y_train)
# print(grid_search.best_score_)
# print(grid_search.best_params_)


#Getting Prediction with ideal parameters
rf = RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=300)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, rf_pred)
print(accuracy)

#Area under ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, rf_pred)
auc = roc_auc_score(y_test, rf_pred)
                    
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
