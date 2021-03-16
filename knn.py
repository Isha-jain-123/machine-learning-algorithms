# THE FOLLOWING MODEL HAS BEEN DESIGNED TO PREDICT THE NATURE OF TUMOUR (BENIGN OR MALLIGNANT) IN A GIVEN SAMPLE

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors,svm

df=pd.read_csv("cancer_dataset1.data")
df.replace('?',-99999,inplace=True) # here we could also instead do df.dropna(inplace=True)
df.drop(["id"],1,inplace=True)
print(df)

X=np.array(df.drop(["class"],1))
y=np.array(df["class"])

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print("The accuracy of our prediction is :",accuracy)

# making a prediction
example_measures=np.array([[4,2,3,1,2,3,4,2,1],[8,2,4,2,2,3,4,2,1]])
example_measures=example_measures.reshape(len(example_measures),-1)
prediction=clf.predict(example_measures)
for i in range(len(prediction)):
    if prediction[i]==2:
        print('The tumour is benign for the sample :',example_measures[i])
    else:
        print("The tumour is malignant for the sample :",example_measures[i])
#print(prediction)