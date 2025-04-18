import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
column_names = ['letter', 'x-box', 'y-box', 'width', 'high', 'onpix',
                'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybar',
                'xy2bar', 'x-ege', 'xegvy', 'y-ege', 'yegvx']

df = pd.read_csv(url, header=None, names=column_names)

#Exploring the data
print(df.head())
print(df.info())
print(df.describe())
input("Press enter to continue...")

print("Missing values: \n", df.isnull().sum())
df['letter'] = df['letter'].astype('category')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['letter'])

print(df[['letter', 'label_encoded']].drop_duplicates().sort_values(by='label_encoded'))
input('Press enter to continue...')

#Frequency of each letter
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='letter', order=sorted(df['letter'].unique()))
plt.title("Letter Frequency")
plt.show()

#correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.drop('letter', axis=1).corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

#Boxplot of a feature
plt.figure(figsize=(12, 5))
sns.boxplot(x='letter', y='x-bar', data=df)
plt.title('x-bar distribution by letter')
plt.show()

#model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X=df.drop(['letter', 'label_encoded'], axis=1)
y=df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=False, cmap="Blues")
plt.title("Confusion matrix")
plt.show()

joblib.dump(model, 'letter_classifier.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.joblib')
