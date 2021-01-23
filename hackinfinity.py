import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , Normalizer
import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

if __name__== "__main__":
    df=pd.read_csv(r'E:/output_3.csv')
    df.loc[df['fever']<99,'fever'] = 0
    df.loc[df['fever']>=99,'fever'] = 1
    data=df.copy()
    data.drop('age',inplace=True,axis=1)
    data.drop('temp',axis=1,inplace=True)
    data['sum1']=0
    for i in data.columns:
        data['sum1']+=data[i]
    data['sum1'].mean()
    print(data['sum1'])
    data.loc[data['sum1']<12,'sum1']=0
    data.loc[data['sum1']>=12,'sum1']=1
    data['infectionProb']*=2
    data['infectionProb'].unique()
    data['finalProb'] = data['infectionProb'] + data['sum1']
    df['infectionProb']=data['finalProb']

    scaled_features = df.copy()
    col_names = ['fever', 'age']
    features = scaled_features[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features[col_names] = features
    print(scaled_features)
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2  #target column i.e price range
#apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=7)
    x=df.drop(['infectionProb','temp'],axis=1)
    y=df['infectionProb']
    fit = bestfeatures.fit(x,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3)
    model=RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1)
    model.fit(x_train,y_train)


    y_pred=model.predict(x_test)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test,y_pred)
    import pickle
    file = open('model.pkl','wb')
    pickle.dump(model, file)
    file.close()