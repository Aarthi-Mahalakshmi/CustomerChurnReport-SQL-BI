#!/usr/bin/env python
# coding: utf-8

# # customer churn 

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
from imblearn.over_sampling import SMOTE
import joblib


# In[2]:


# load the training data
data = pd.read_excel("Prediction_Data.xlsx", sheet_name="vw_churndata")
data.head()

data preprocessing
# In[3]:


# dropping the columns that wont be used for predictions
data = data.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)


# In[4]:


#List of columns to be label encoded
columns_to_encode = ['Gender', 'Married', 'State', 'Value_Deal', 'Phone_Service', 'Multiple_Lines',
    'Internet_Service', 'Internet_Type', 'Online_Security', 'Online_Backup',
    'Device_Protection_Plan', 'Premium_Support', 'Streaming_TV', 'Streaming_Movies',
    'Streaming_Music', 'Unlimited_Data', 'Contract', 'Paperless_Billing',
    'Payment_Method']
   


# In[5]:


# Encode categorical variables except the target variable
label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])


# In[6]:


# Manually encode the target variable 'Customer_Status'
data['Customer_Status'] = data['Customer_Status'].map({'Stayed': 0, 'Churned': 1})


# In[7]:


# Split data into features and target
X = data.drop('Customer_Status', axis=1)
y = data['Customer_Status']


# In[ ]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # For ROC curve and AUC score

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    return conf_matrix, report, fpr, tpr, roc_auc, model.feature_importances_

Random Forest
# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# train the model
rf_model.fit(X_train, y_train)


# In[12]:


# Make predictions
y_pred = rf_model.predict(X_test)


# In[25]:


# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
class_report_rf = classification_report(y_test, y_pred, output_dict=True)

Random forest(applying smote)
# In[14]:


# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[15]:


# Split resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[16]:


# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[27]:


# Evaluate the model
y_pred = rf_model.predict(X_test)
print("Confusion Matrix after addressing class imbalance:")
print(confusion_matrix(y_test, y_pred))
conf_matrix_rf_smote = confusion_matrix(y_test, y_pred)
print("\nClassification Report after addressing class imbalance:")
print(classification_report(y_test, y_pred))
class_report_rf_smote = classification_report(y_test, y_pred, output_dict=True)

XG Boost
# In[18]:


xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)


# In[28]:


# Evaluate the model
y_pred = xgb_model.predict(X_test)
print("Confusion Matrix with XGBoost:")
print(confusion_matrix(y_test, y_pred))
conf_matrix_xgb = confusion_matrix(y_test, y_pred)
print("\nClassification Report with XGBoost:")
print(classification_report(y_test, y_pred))
class_report_xgb = classification_report(y_test, y_pred, output_dict=True)


# In[23]:


# Plot Confusion Matrix Heatmap
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

plot_confusion_matrix(conf_matrix_rf, "Random Forest without SMOTE")
plot_confusion_matrix(conf_matrix_rf_smote, "Random Forest with SMOTE")
plot_confusion_matrix(conf_matrix_xgb, "XGBoost")


# In[29]:


# Compare Models and Select the Best One
def get_best_model(report_rf, report_rf_smote, report_xgb):
    models = {
        "Random Forest without SMOTE": report_rf,
        "Random Forest with SMOTE": report_rf_smote,
        "XGBoost": report_xgb
    }
    best_model = max(models, key=lambda k: models[k]['accuracy'])
    return best_model

best_model_name = get_best_model(class_report_rf, class_report_rf_smote, class_report_xgb)
print(f"The best model is: {best_model_name}")


# # Conclusion
# Given that all models have the same overall accuracy and very similar precision, recall, and F1-scores
# Overall, Random Forest with SMOTE might edge out slightly due to its balanced results. 
# However, any of the three models could be considered suitable given their similar performance.

# In[ ]:


# Load the Best Model for Prediction
if best_model_name == "Random Forest without SMOTE":
    best_model = joblib.load('rf_model_no_smote.pkl')
elif best_model_name == "Random Forest with SMOTE":
    best_model = joblib.load('rf_model_smote.pkl')
else:
    best_model = joblib.load('xgb_model.pkl')


# In[ ]:


#Use Model for Prediction on New Data


# In[31]:


# load the predicting data
df = pd.read_excel("Prediction_Data.xlsx", sheet_name="vw_joindata")
df.head()


# In[36]:


# Retain the original DataFrame to preserve unencoded columns
original_df = df.copy()


# In[37]:


# Retain the Customer_ID column
customer_ids = df['Customer_ID']


# In[38]:


# Drop columns that won't be used for prediction in the encoded DataFrame
new_data = df.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason'], axis=1)


# In[39]:


# Encode categorical variables using the saved label encoders
for column in new_data.select_dtypes(include=['object']).columns:
    new_data[column] = label_encoders[column].transform(new_data[column])


# In[40]:


# Make predictions
new_predictions = rf_model.predict(new_data)


# In[42]:


# Add predictions to the original DataFrame
original_df['Customer_Status_Predicted'] = new_predictions


# In[43]:


# Filter the DataFrame to include only records predicted as "Churned"
original_df = original_df[original_df['Customer_Status_Predicted'] == 1]


# In[45]:


# Save the results
original_df.to_csv('Predictions.csv', index=False)


# In[ ]:




