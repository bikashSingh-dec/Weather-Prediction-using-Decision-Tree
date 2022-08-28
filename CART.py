#Gini index
from sklearn import tree
clf=tree.DecisionTreeClassifier(criterion='gini')
clf=clf.fit(x_train,y_train)
import graphviz
dot_data=tree.export_graphviz(clf, out_file=None)
graph=graphviz.Source(dot_data)
graph
