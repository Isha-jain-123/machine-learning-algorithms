import pandas as pd
import numpy as np
from sklearn import preprocessing,model_selection,neighbors,svm

df=pd.read_csv("ssds.csv")
df=df.drop(['objid','specobjid','redshift','plate','fiberid'],axis=1)
df["class"]=df["class"].astype("category").cat.codes #converted stars and galaxies etc. to numbers
#print(df.shape)
#print(df)
#star=2 , galaxy=0 , qso=1

#now we will scale the data
'''
x=df.values
min_max_scaler=preprocessing.MinMaxScaler()
scaled=min_max_scaler.fit_transform(x)
df=pd.DataFrame(scaled)
'''
min_max_scaler=preprocessing.MinMaxScaler()
df[['ra','dec','u','g','r','i','z','run','rerun','camcol','field','mjd']]=min_max_scaler.fit_transform(df[['ra','dec','u','g','r','i','z','run','rerun','camcol','field','mjd']])


#creating the model
X=np.array(df.drop(["class"],1))
y=np.array(df["class"])

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print("The accuracy of our prediction is :",accuracy)

#making a prediction
example_measures=np.array([[0.19,0.12,0.17,0.18,0.16,0.17,0.15,0.7,0.3,0.4,0.26,0.5]])
example_measures=example_measures.reshape(len(example_measures),-1)
prediction=clf.predict(example_measures)
#print(prediction)