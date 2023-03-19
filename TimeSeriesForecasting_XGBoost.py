''' Time Series Forecasting with XGBoost - Use Python and ML to predict energy consumption '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

### Data Preparation

path = 'PJME_hourly.csv'
df = pd.read_csv(path)
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

df.plot(style='-',
        figsize=(10, 5),
        color=color_pal[0],
        title='PJME Energy Use in MW')
plt.xticks( rotation = 45)
plt.show()

### Train/Test Split

train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

fig,ax = plt.subplots(figsize = (10,5))
train.plot(ax = ax, label = 'Training Set')
test.plot(ax = ax, label = 'Test Set')
plt.legend(labels=['Training Set','Test Set'], loc = 'upper right')
plt.title('Train/Test Split')
ax.axvline('01-01-2015', color = 'red', ls = '--')
plt.show()

#Single week data analysis

df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')].plot(figsize = (10,5),
                                                                   title = 'Week of Data')
plt.show()

### Feature Creation
def create_features(df):
    '''
    Create time series features based on time series index
    '''
    df = df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['quarter'] = df.index.quarter
    return df
df = create_features(df)

#Visualize our feature to target relationship

fig,ax = plt.subplots(figsize=(10,5))
sns.boxplot(data=df, x='hour', y='PJME_MW')
ax.set_title('MW by Hour')
plt.show()

fig,ax = plt.subplots(figsize=(10,5))
sns.boxplot(data=df, x='quarter', y='PJME_MW')
ax.set_title('MW by quarter')
plt.show()

fig,ax = plt.subplots(figsize=(10,5))
sns.boxplot(data=df, x='month', y='PJME_MW')
ax.set_title('MW by month')
plt.show()

### Create Model -> XGB Regressor

train = create_features(train)
test = create_features(test)

FEATURES = ['hour','dayofweek','quarter',
            'month', 'year', 'dayofyear']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train,y_train,
        eval_set = [(X_train,y_train),(X_test,y_test)],
        verbose=True)

### Feature Importance

fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh',title='Feature Importance')
plt.show()

### Forecast on Test

'''Year Prediction'''
test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)

ax = df[['PJME_MW']].plot(figsize=(10,5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data','Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()

'''Week Prediction'''
ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'] \
    .plot(figsize=(15, 5), title='Week Of Data')
df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'] \
    .plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()

### Metric: RMSE

score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

### Calculate Error

#Look at worst and best predicted days
test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)

