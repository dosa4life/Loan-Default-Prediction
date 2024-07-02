import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import imblearn

df_ = pd.read_csv(r"D:\UTA\Spring_2024\Data Minning\Project\Data Mining Project\Training Data.csv")
df = df_.copy()
df.shape


def clean_and_encode_location(df):
    df['CITY_'] = df['CITY'].replace(r'[\d+\[\]]','',regex=True)
    df['STATE_'] = df['STATE'].str.replace('_',' ')
    df['Location'] = df['CITY_']+df['STATE_']
    df['Location'] = df['Location'].replace('\s+','',regex=True)
    df['Location'] = df['Location'].replace('[,_-]+','',regex=True)
    df['Location'] = df['Location'].replace(r'[\d+\[\]]','',regex=True)
    df['Location'] = df['Location'].replace(r'[^a-zA-Z]','',regex=True) 
    df['Location'] = df['Location'].str.lower()
    return df

def enocoding_binary_columns(df):
    df['enc_Married_Single'] = df['Married/Single'].map({'single' : 1, 'married':0})
    df['enc_car_ownership'] = df['Car_Ownership'].map({'no':0, 'yes':1})
    return df

def BNRY_ENCDR(df,col):
    encoder = ce.BinaryEncoder(cols=[col],return_df=True)
    df = encoder.fit_transform(df) 
    return df

def OHE_ENCDR(df,col):
    encoder=ce.OneHotEncoder(cols=col,handle_unknown='return_nan',return_df=True,use_cat_names=True)
    df = encoder.fit_transform(df)
    return df

def FX_ENCDR(df,col):
    enc_fx = (df.groupby(col).size()) / len(df)
    enc_col = 'enc_' + col
    df[enc_col] = df[col].apply(lambda x : enc_fx[x])
    return df

def TRG_ENCDR(df,col,trg):
    encoder_=ce.TargetEncoder(cols=col)
    encoder_.fit_transform(df[col],df[trg])
    enc_col = 'enc_' + col
    df[enc_col] = encoder_.transform(df[col])
    return df

def LBL_ENCDR(df,col):
    le = LabelEncoder()
    enc_col = 'enc_' + col
    df[enc_col] = le.fit_transform(df[col])
    return df
	
_df = clean_and_encode_location(df)
# _df = enocoding_binary_columns(_df)
_df = OHE_ENCDR(_df,'Married/Single')
_df = OHE_ENCDR(_df,'Car_Ownership')
# _df = OHE_ENCDR(_df,'House_Ownership')
_df = BNRY_ENCDR(_df,'House_Ownership')
# _df = TRG_ENCDR(_df,'Profession','Risk_Flag')
# _df = FX_ENCDR(_df,'Profession')
_df = LBL_ENCDR(_df,'Profession')
# _df = FX_ENCDR(_df,'Location')
_df = LBL_ENCDR(_df,'Location')
# _df = TRG_ENCDR(_df,'Location','Risk_Flag')
_df.drop(['Id','CITY', 'STATE'],axis=1,inplace=True)
# _df.drop_duplicates(inplace=True)
_df.shape

def STD_SCALING(df,col):
    scaler = StandardScaler()
    enc_col = 'enc_' + col
    df[enc_col] = scaler.fit_transform(df[[col]])
    return df
	
_df = STD_SCALING(_df,'Income')
_df = STD_SCALING(_df,'Age')
_df = STD_SCALING(_df,'Experience')
_df = STD_SCALING(_df,'CURRENT_JOB_YRS')
_df = STD_SCALING(_df,'CURRENT_HOUSE_YRS')
_df = STD_SCALING(_df,'enc_Profession')
_df = STD_SCALING(_df,'enc_Location')
_df.loc[:,['enc_Income', 'enc_Age', 'enc_Experience','enc_CURRENT_JOB_YRS', 'enc_CURRENT_HOUSE_YRS','enc_enc_Profession', 'enc_enc_Location']].describe().apply(lambda s: s.apply('{0:.5f}'.format))


_df.drop([ 'Income', 'Age', 'Experience','Profession','enc_Profession','CURRENT_JOB_YRS','CURRENT_HOUSE_YRS',
          'CITY_', 'STATE_','Location', 'enc_Location'],axis=1,inplace =True)
		  
_train = _df.sample(frac = 0.8,random_state = 40).copy()
_train.shape

_test = _df.loc[_df.index.isin([i for i in set([i for i in _df.index]) - set([i for i in _train.index])]),: ].copy()
_test.shape

X = _train.copy()
X.drop('Risk_Flag',axis=1,inplace=True)
y = _train.loc[:,['Risk_Flag']].copy()

smt = SMOTE()
print(f"Before Sampling: {y.value_counts()}")
X_smt, y_smt = smt.fit_resample(X, y)
print(f"After using SMOTE Sampling: {y_smt.value_counts()}")

ada = ADASYN(random_state=130)
print(f"Before Sampling: {y.value_counts()}")
X_ada, y_ada = ada.fit_resample(X, y)
print(f"After ADASYN Sampling: {y_ada.value_counts()}")

oversample = RandomOverSampler(sampling_strategy=0.5)
print(f"Before Sampling: {y.value_counts()}")
X_os, y_os = oversample.fit_resample(X, y)
print(f"After Sampling: {y_os.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(_df.loc[:,['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 
        'CURRENT_HOUSE_YRS', 'enc_Married_Single', 'enc_car_ownership',
       'enc_House_Ownership', 'enc_Profession', 'enc_Location']], _df.loc[:,['Risk_Flag']], test_size=0.2, random_state=1)
##DT: 0.6621; Log_Reg: 0.8095;    SVM: 0.8095


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
#DT: 0.6621; Log_Reg: 0.8095;    SVM: 0.8095

print("SMOTE")
for i in range(0,1):
    X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt, test_size=0.2)
    for name, model in models:
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate Metric
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        #Confusion Matrixs
        disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
#         display_labels=class_names,
        cmap=plt.cm.Blues,
#         normalize=normalize,
        )
        disp.ax_.set_title(f"{name} Confusion Matrix")

        # Print model name and accuracy
        print(f"Iteration {i}: {name}:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")

print("ADASYN")
for i in range(0,1):
    X_train, X_test, y_train, y_test = train_test_split(X_ada, y_ada, test_size=0.2)
    for name, model in models:
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate Metric
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        #Confusion Matrixs
        disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
#         display_labels=class_names,
        cmap=plt.cm.Blues,
#         normalize=normalize,
        )
        disp.ax_.set_title(f"{name} Confusion Matrix")

        # Print model name and accuracy
        print(f"Iteration {i}: {name}:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
		
		
# print("Random Over Sampling")
for i in range(0,1):
    X_train, X_test, y_train, y_test = train_test_split(X_os, y_os, test_size=0.2)
    for name, model in models:
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate Metric
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        #Confusion Matrixs
        disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
#         display_labels=class_names,
        cmap=plt.cm.Blues,
#         normalize=normalize,
        )
        disp.ax_.set_title(f"{name} Confusion Matrix")

        # Print model name and accuracy
        print(f"Iteration {i}: {name}:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")

# Feature Selection
smote_RF = RandomForestClassifier(random_state=42).fit(X_smt, y_smt)

importances = smote_RF.feature_importances_
features = X_smt.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('SMOTE + RF: Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt, test_size=0.2)
# smote_RF = RandomForestClassifier(random_state=42).fit(X_train, y_train)
X_train.shape, X_test.shape
X_smt.shape
X.shape

X_smt_  = X_smt.drop(['House_Ownership_0','House_Ownership_1','Car_Ownership_no','Car_Ownership_yes',
                      'Car_Ownership_no','Married/Single_single', 'Married/Single_married'],axis=1).copy()
X_train, X_test, y_train, y_test = train_test_split(X_smt_, y_smt, test_size=0.2)
smote_RF = RandomForestClassifier(random_state=42).fit(X_train, y_train)

y_pred_smote_RF = smote_RF.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_smote_RF)
precision = precision_score(y_test, y_pred_smote_RF)
recall = recall_score(y_test, y_pred_smote_RF)
f1 = f1_score(y_test, y_pred_smote_RF)
roc_auc = roc_auc_score(y_test, y_pred_smote_RF)
print('\nPredicting using Test data with SMOTE + Random Forrest Classifier\n')
print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
cm = confusion_matrix(y_test, y_pred_smote_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for SMOTE + Random Forest Classifier')
plt.show()

X_smt_  = X_smt.drop(['House_Ownership_0','House_Ownership_1','Car_Ownership_no','Car_Ownership_yes',
                      'Car_Ownership_no','Married/Single_single', 'Married/Single_married'],axis=1).copy()
smote_RF = RandomForestClassifier(random_state=42).fit(X_smt_, y_smt)

X_test = _test.copy()
X_test.drop(['House_Ownership_0','House_Ownership_1','Car_Ownership_no','Car_Ownership_yes',
                        'Car_Ownership_no','Married/Single_single', 'Married/Single_married','Risk_Flag'],
            axis=1,inplace=True)
y_test = _test.loc[:,['Risk_Flag']].copy()

y_pred_smote_RF = smote_RF.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_smote_RF)
precision = precision_score(y_test, y_pred_smote_RF)
recall = recall_score(y_test, y_pred_smote_RF)
f1 = f1_score(y_test, y_pred_smote_RF)
roc_auc = roc_auc_score(y_test, y_pred_smote_RF)
print('\nPredicting using Unseen data with SMOTE + Random Forrest Classifier (Important Columns)\n')
print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
cm = confusion_matrix(y_test, y_pred_smote_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for SMOTE + Random Forest Classifier')
plt.show()


adasyn_RF = RandomForestClassifier(random_state=42).fit(X_ada, y_ada)

importances = adasyn_RF.feature_importances_
features = X_ada.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('ADASYN + RF: Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

X_ada_  = X_ada.drop(['House_Ownership_0','House_Ownership_1','Car_Ownership_no','Car_Ownership_yes',
                      'Car_Ownership_no','Married/Single_single', 'Married/Single_married'],axis=1).copy()
X_train, X_test, y_train, y_test = train_test_split(X_ada_, y_ada, test_size=0.2)
adasyn_RF = RandomForestClassifier(random_state=42).fit(X_train, y_train)

y_pred_ada_RF = adasyn_RF.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_ada_RF)
precision = precision_score(y_test, y_pred_ada_RF)
recall = recall_score(y_test, y_pred_ada_RF)
f1 = f1_score(y_test, y_pred_ada_RF)
roc_auc = roc_auc_score(y_test, y_pred_ada_RF)
print('\nPredicting using Test data with ADASYN + Random Forrest Classifier\n')
print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
cm = confusion_matrix(y_test, y_pred_ada_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for ADASYN + Random Forest Classifier')
plt.show()

X_ada_  = X_ada.drop(['House_Ownership_0','House_Ownership_1','Car_Ownership_no','Car_Ownership_yes',
                      'Car_Ownership_no','Married/Single_single', 'Married/Single_married'],axis=1).copy()
adasyn_RF = RandomForestClassifier(random_state=42).fit(X_ada_, y_ada)

X_test = _test.copy()
X_test.drop(['House_Ownership_0','House_Ownership_1','Car_Ownership_no','Car_Ownership_yes',
                        'Car_Ownership_no','Married/Single_single', 'Married/Single_married','Risk_Flag'],
            axis=1,inplace=True)
y_test = _test.loc[:,['Risk_Flag']].copy()

y_pred_adasyn_RF = adasyn_RF.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_adasyn_RF)
precision = precision_score(y_test, y_pred_adasyn_RF)
recall = recall_score(y_test, y_pred_adasyn_RF)
f1 = f1_score(y_test, y_pred_adasyn_RF)
roc_auc = roc_auc_score(y_test, y_pred_adasyn_RF)
print('\nPredicting using Unseen data with ADASYN + Random Forrest Classifier  (Important Columns)\n')
print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
cm = confusion_matrix(y_test, y_pred_adasyn_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for ADASYN + Random Forest Classifier')
plt.show()

ros_RF = RandomForestClassifier(random_state=42).fit(X_os, y_os)

importances = ros_RF.feature_importances_
features = X_os.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('ROS + RF: Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

X_os_  = X_os.drop(['House_Ownership_0','House_Ownership_1','Car_Ownership_no','Car_Ownership_yes',
                      'Car_Ownership_no','Married/Single_single', 'Married/Single_married'],axis=1).copy()
X_train, X_test, y_train, y_test = train_test_split(X_os_, y_os, test_size=0.2)
ros_RF = RandomForestClassifier(random_state=42).fit(X_train, y_train)

y_pred_ros_RF = ros_RF.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_ros_RF)
precision = precision_score(y_test, y_pred_ros_RF)
recall = recall_score(y_test, y_pred_ros_RF)
f1 = f1_score(y_test, y_pred_ros_RF)
roc_auc = roc_auc_score(y_test, y_pred_ros_RF)
print('\nPredicting using Test data with ROS + Random Forrest Classifier\n')
print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
cm = confusion_matrix(y_test, y_pred_ros_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for ROS + Random Forest Classifier')
plt.show()

X_os_  = X_os.drop(['House_Ownership_0','House_Ownership_1','Car_Ownership_no','Car_Ownership_yes',
                      'Car_Ownership_no','Married/Single_single', 'Married/Single_married'],axis=1).copy()
ros_RF = RandomForestClassifier(random_state=42).fit(X_os_, y_os)

X_test = _test.copy()
X_test.drop(['House_Ownership_0','House_Ownership_1','Car_Ownership_no','Car_Ownership_yes',
                        'Car_Ownership_no','Married/Single_single', 'Married/Single_married','Risk_Flag'],
            axis=1,inplace=True)
y_test = _test.loc[:,['Risk_Flag']].copy()

y_pred_ros_RF = ros_RF.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_ros_RF)
precision = precision_score(y_test, y_pred_ros_RF)
recall = recall_score(y_test, y_pred_ros_RF)
f1 = f1_score(y_test, y_pred_ros_RF)
roc_auc = roc_auc_score(y_test, y_pred_ros_RF)
print('\nPredicting using Unseen data with ROS + Random Forrest Classifier (Important Columns)\n')
print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
cm = confusion_matrix(y_test, y_pred_ros_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for ROS + Random Forest Classifier')
plt.show()

smote_RF = RandomForestClassifier(random_state=42).fit(X_smt, y_smt)
adasyn_RF = RandomForestClassifier(random_state=42).fit(X_ada, y_ada)
ros_RF = RandomForestClassifier(random_state=42).fit(X_os, y_os)

X_test = _test.copy()
X_test.drop('Risk_Flag',axis=1,inplace=True)
y_test = _test.loc[:,['Risk_Flag']].copy()

y_pred_smote_RF = smote_RF.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_smote_RF)
precision = precision_score(y_test, y_pred_smote_RF)
recall = recall_score(y_test, y_pred_smote_RF)
f1 = f1_score(y_test, y_pred_smote_RF)
roc_auc = roc_auc_score(y_test, y_pred_smote_RF)
print('\nPredicting using Unseen data with SMOTE + Random Forrest Classifier (All Columns)\n')
print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
cm = confusion_matrix(y_test, y_pred_smote_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for SMOTE + Random Forest Classifier')
plt.show()


adasyn_RF = RandomForestClassifier(random_state=42).fit(X_ada, y_ada)
y_pred_adasyn_RF = adasyn_RF.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_adasyn_RF)
precision = precision_score(y_test, y_pred_adasyn_RF)
recall = recall_score(y_test, y_pred_adasyn_RF)
f1 = f1_score(y_test, y_pred_adasyn_RF)
roc_auc = roc_auc_score(y_test, y_pred_adasyn_RF)
print('\nPredicting using Unseen data with ADASYN + Random Forrest Classifier (All Columns)\n')
print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
cm = confusion_matrix(y_test, y_pred_adasyn_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for ADASYN + Random Forest Classifier')
plt.show()

y_pred_ros_RF = ros_RF.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_ros_RF)
precision = precision_score(y_test, y_pred_ros_RF)
recall = recall_score(y_test, y_pred_ros_RF)
f1 = f1_score(y_test, y_pred_ros_RF)
roc_auc = roc_auc_score(y_test, y_pred_ros_RF)
print('\nPredicting using Unseen data with Random Over Sampling + Random Forrest Classifier (All Columns)\n')
print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nROC_AUC: {roc_auc:.2f}\n")
cm = confusion_matrix(y_test, y_pred_ros_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for Random Over Sampling + Random Forest Classifier')
plt.show()