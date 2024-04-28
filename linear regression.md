## Logistic Regression 
This is my ATM Project (Amati (observe), Tiru (imitate), Modifikasi (modification))\
Learn into ML DL AI as the beginner to be a hero!

### Import Libraries


```python
import pandas as pd # data manipulation
import numpy as np # array manipulation
%matplotlib inline 
import matplotlib.pyplot as plt # visualization
import sklearn
from sklearn.compose import ColumnTransformer # transform the column 
from sklearn.preprocessing import OneHotEncoder # encode datatype
from sklearn.linear_model import LinearRegression # modelling
from sklearn.model_selection import train_test_split # splitting train and test
from sklearn.metrics import mean_squared_error # evaluation
```

### Import Dataset


```python
dt = pd.read_csv("data_input/insurance.csv")
```


```python
dt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>female</td>
      <td>27.900</td>
      <td>0</td>
      <td>yes</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>male</td>
      <td>33.770</td>
      <td>1</td>
      <td>no</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>male</td>
      <td>33.000</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>male</td>
      <td>22.705</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>male</td>
      <td>28.880</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>




```python
dt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1338 entries, 0 to 1337
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1338 non-null   int64  
     1   sex       1338 non-null   object 
     2   bmi       1338 non-null   float64
     3   children  1338 non-null   int64  
     4   smoker    1338 non-null   object 
     5   region    1338 non-null   object 
     6   charges   1338 non-null   float64
    dtypes: float64(2), int64(2), object(3)
    memory usage: 73.3+ KB
    

In this step, we importing the dataset from data_input folder. This data has 1071 rows with 7 columns. The data type is still not relevan, so we turn it into the right types. 

### Data Wrangling

In this step, we would check if there is any missing values, unappropriate data types, duplicates, etc.


```python
dt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1337 entries, 0 to 1337
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1337 non-null   int64  
     1   sex       1337 non-null   object 
     2   bmi       1337 non-null   float64
     3   children  1337 non-null   int64  
     4   smoker    1337 non-null   object 
     5   region    1337 non-null   object 
     6   charges   1337 non-null   float64
    dtypes: float64(2), int64(2), object(3)
    memory usage: 83.6+ KB
    


```python
dt[['sex','smoker','region']] = dt[['sex','smoker','region']].astype('category')
```

##### Missing value


```python
dt.isnull().any()
```




    age         False
    sex         False
    bmi         False
    children    False
    smoker      False
    region      False
    charges     False
    dtype: bool




```python
dt.duplicated().any()
```




    True




```python
dt.drop_duplicates(inplace = True)
```


```python
dt.duplicated().any()
```




    False



### Splitting Data

Split data into target and its predictors.


```python
# get the target
y = dt['charges']

# drop target to get predictors
X = dt[['age','children']]
```


```python
# split 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, \
                                                    y,\
                                                    test_size=0.2,\
                                                    random_state=42)
```

### Training

Fitting data using train that has been splitted in previous step. We can use LinearRegression function from sklearn.


```python
mdl = LinearRegression()
mdl.fit(X_train, y_train)
```




    LinearRegression()



### Testing

Predict model that has builded with test data


```python
y_pred = mdl.predict(X_test)
y_pred[:5]
```




    array([14630.08887853, 12588.95412289, 17160.86493902, 12382.48008346,
           11698.16755848])



### Model Evaluation


```python
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error : ", mse)
```

    Mean Squared Error :  166847445.41074288
    

Our model has 16% error which can tell that our model is quite good for predicting new data. Next, we predict new (dummy) data


```python
new_dt = {'age' : [50,30,20],
         'children' : [7, 0, 4]}

new_dt = pd.DataFrame(new_dt)
```


```python
y_hat = mdl.predict(new_dt)
y_hat
```




    array([19659.19579616, 10090.62326575, 10438.68093789])



Our model success to predict new data. If you have any advice for this work or suggestion what i should learn next, you can [send an email](mailto:rahfairuzran@gmail.com) to
 me
