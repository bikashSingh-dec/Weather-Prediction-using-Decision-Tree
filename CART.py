#importing Libraries
import numpy as np
import pandas as pd
df = pd.read_csv('data.csv')

#get dummy columns for equivalent numerical values
def_getdummy=pd.get_dummies(data=df, column=['temp','outlook','windy','humidity'])

#import train_test function from sklearn library
from sklearn.model_selection import train_test_split
x=df_getdummy.drop('play',axis=1)
y=df_getdummy['play']
x_train, x_test, y_train, y_test = train_test_split(x , y , test.size=0.3, random_state=101)

#importing decision tree classifier from sklearn and fit the model
from sklearn.tree import DecisionTreeClassifier
dtree=DesicionTreeClassifier(max_depth=3)
dtree.fit(x_train, y_train)
prediction = dtree.predict(x_test)

#Gini index
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
clf=tree.DecisionTreeClassifier(criterion='gini')
clf=clf.fit(x_train,y_train)
import graphviz
dot_data=tree.export_graphviz(clf, out_file=None)
graph=graphviz.Source(dot_data)
graph
