<H3>ENTER YOUR NAME : CHARU NETHRA R
<H3>ENTER YOUR REGISTER NO.: 212223230035
<H3>EX. NO.1</H3>
<H3>DATE : 07/03/2025
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


## OUTPUT:
### Dataset:
![OP1](https://github.com/user-attachments/assets/322a482e-02cb-4cfa-95bf-3d1d67eb9383)

### X Values:
![OP2](https://github.com/user-attachments/assets/343634f9-4c49-4b19-90a7-36054b6750bd)

### Y Values:
![OP3](https://github.com/user-attachments/assets/d2ac58b7-158e-42cd-84c8-cd7247193c2e)

### Null Values:
![OP4](https://github.com/user-attachments/assets/65729c4a-8669-4206-916c-0edcdf68ed02)

### Duplicated Values:
![OP5](https://github.com/user-attachments/assets/dec4109a-d3fe-4365-8259-17b24f32d44b)

### Description:
![OP6](https://github.com/user-attachments/assets/22b5c424-026f-4f60-8be1-3af700c99574)

### Normalized Dataset: 
![OP8](https://github.com/user-attachments/assets/16523645-f2da-41c0-833e-f2545979ef92)

### Training Data:
![OP9](https://github.com/user-attachments/assets/02cfc404-2ff6-40af-8e4d-91814822abfc)

### Testing Data:
![OP10](https://github.com/user-attachments/assets/e3203309-e796-4cc7-a97d-66a32b4f8b54)

### Output:
![OP11](https://github.com/user-attachments/assets/cec439e3-bacf-4556-8674-c118733e97e6)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


