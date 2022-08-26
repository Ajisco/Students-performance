from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from treeinterpreter import treeinterpreter as ti 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

app= Flask(__name__)

# Loading Files, df_mat for maths, df_por for portuguese
df_mat =pd.read_csv('student-mat.csv',sep=';')
df_por =pd.read_csv('student-por.csv',sep=';')
# Merging both datframes
df=pd.concat([df_mat, df_por], ignore_index=True)
# Create Annual Average column as 'avg'
columns = ['G1', 'G2', 'G3']
df['avg'] = df[columns].mean(axis=1)
# Classifies scores using the 5-Level Classification
'''
(0, 9.5] -- Fail
(9.5, 11.5] -- Sufficient/ Fair
(11.5, 13.5] -- Satisfactory
(13.5, 15.5] -- good
(15.5, 20] -- Excellent
'''


bins = pd.IntervalIndex.from_tuples(
    [(0, 9.5), (9.5, 11.5), (11.5, 13.5), (13.5, 15.5), (15.5, 20)], closed='right')

levels = ['fail', 'sufficient', 'satisfactory', 'good', 'excellent']

new_column = 'grades'
df[new_column] = np.array(levels)[pd.cut(df['avg'], bins=bins).cat.codes]

# Outliers

def detect_outliers(columns):
    outlier_indices = []
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.

        mask = (df[column] >= Q1 - 1.5 *IQR) & (df[column] <= Q3 + 1.5 * IQR)
        mask = mask.to_numpy()
        false_indices = np.argwhere(~mask)
        outlier_indices.append(false_indices)
    return np.unique(np.concatenate(outlier_indices).ravel())

numerical_columns = ['age', 'absences', 'avg']
outlier_indices = detect_outliers(numerical_columns)

# Delete outliers
df = df.drop(outlier_indices, axis=0)

df.drop(['G1', 'G2', 'G3', 'avg', 'school'], inplace = True, axis = 1)
# Split dataset
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size=0.25, random_state=42)

# Data Scaling

# First we need to know which columns are binary, nominal and numerical
def get_columns_by_category():
    categorical_mask = X.select_dtypes(
        include=['object']).apply(pd.Series.nunique) == 2
    numerical_mask = X.select_dtypes(
        include=['int64', 'float64']).apply(pd.Series.nunique) > 5

    binary_columns = X[categorical_mask.index[categorical_mask]].columns
    nominal_columns = X[categorical_mask.index[~categorical_mask]].columns
    numerical_columns = X[numerical_mask.index[numerical_mask]].columns

    return binary_columns, nominal_columns, numerical_columns

binary_columns, nominal_columns, numerical_columns = get_columns_by_category()

# Now we can create a column transformer pipeline

transformers = [('binary', OrdinalEncoder(), binary_columns),
                ('nominal', OneHotEncoder(), nominal_columns),
                ('numerical', StandardScaler(), numerical_columns)]

transformer_pipeline = ColumnTransformer(transformers, remainder='passthrough')

# Starified k cross validation
Kfold = StratifiedKFold(n_splits=5)

model = RandomForestClassifier(max_depth=7, 
                       min_samples_split=5, 
                       min_samples_leaf=5, random_state=42)

pipe = Pipeline([('transformer', transformer_pipeline), ('Random Forest Classifier', model)])

# Cross Validation

def cv_fit_models():
    train_acc_results = []
    cv_scores = {'Random Forest Classifier': []}
    cv_score = cross_validate(pipe,
                              X_train,
                              y_train,
                              scoring=scoring,
                              cv=Kfold,
                              return_train_score=True,
                              return_estimator=True)

    train_accuracy = cv_score['train_acc'].mean() * 100

    train_acc_results.append(train_accuracy)
    cv_scores['Random Forest Classifier'].append(cv_score)

    return np.array(train_acc_results), cv_scores

scoring = {'acc': 'accuracy'}

results, folds_scores = cv_fit_models()

# Pick the best fold for each model according to the highest test accuracy:

def pick_best_estimator():
    best_estimators = {'Random Forest Classifier': []}
    for key, model in folds_scores.items():
        best_acc_idx = np.argmax(model[0]['test_acc'])
        best_model = model[0]['estimator'][best_acc_idx]
        best_estimators[key].append(best_model)
    return best_estimators

best_estimators = pick_best_estimator()

modl =  best_estimators['Random Forest Classifier'][0]

msg =  (
    'The se of the student',
    'The age of the student',
    'Student home area (Urban or Rural)',
    'Family size',
    'Parent staying together or not',
    'Mothers education level',
    'Fathers level of education',
    'Mothers job',
    'Fathers job',
    'Reason for choosing current school',
    'Guardian of the student',
    'Time taken to get to school from home',
    'Hours of study per week',
    'Number of classes previously failed',
    'Access to educational support',
    'Getting family support or not',
    'Attending paid tutorial or not',
    'Extra curricular activities or not',
    'Attended nursery school or not',
    'Plan to attend higher institution',
    'Access to internet at home',
    'Status of romantic relationship',
    'Status of family relationship',
    'Number of free time the student has',
    'How often the student goes out',
    'How often the student takes alcohol in the weekday',
    'How often the student takes alcohol in the weekend',
    'How healthy the student is',
    'How often the student is absent from school'
       )

col_msg = dict(zip(X_train.columns, msg))


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def index():
    sex= request.form['sex']
    age= request.form['age']
    address= request.form['address']
    famsize= request.form['famsize']
    Pstatus= request.form['Pstatus']
    Medu= request.form['Medu']
    Fedu= request.form['Fedu']
    Mjob= request.form['Mjob']
    Fjob= request.form['Fjob']
    reason= request.form['reason']
    guardian= request.form['guardian']
    traveltime= request.form['traveltime']
    studytime= request.form['studytime']
    failures= request.form['failures']
    schoolsup= request.form['schoolsup']
    famsup= request.form['famsup']
    paid= request.form['paid']
    activities= request.form['activities']
    nursery= request.form['nursery']
    higher= request.form['higher']
    internet= request.form['internet']
    romantic= request.form['romantic']
    famrel= request.form['famrel']
    freetime= request.form['freetime']
    goout= request.form['goout']
    Dalc= request.form['Dalc']
    Walc= request.form['Walc']
    health= request.form['health']
    absences= request.form['absences']
    arr = pd.DataFrame((np.array([[sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,
                guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,
                nursery,higher,internet, romantic,famrel,freetime,goout,Dalc,Walc,health,absences]])
        ), columns=X_train.columns)    
    pred= modl.predict(arr)


    prediction, bias, contributions = ti.predict(modl[-1], modl[:-1].transform(arr))
    nums=[]
    for c, feature in sorted(zip(contributions[0], 
                                 X_test.columns), 
                             key=lambda x: ~abs(x[0]).all()):
        rnd=c[np.argmax(bias[0])]
        nums.append(rnd)

    listd = sorted(zip(X_test.columns,nums),key=lambda x: (x[1])**2)

    most = listd[-1][0]

    a,b = list(zip(*listd[-6:-1]))
    next5  = list(reversed(list(a)))
    #col_msg = col_msg


    return render_template('after.html', data=pred ,
        most=most, next5=next5, col_msg = col_msg)
        

if __name__ == '__main__':
    app.run(debug= True, use_reloader=False)
    
    
     




    