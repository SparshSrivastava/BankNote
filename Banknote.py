import numpy as np

banknote=np.loadtxt('data_banknote_authentication.txt',delimiter=',')

import pandas as pd
df=pd.DataFrame(banknote)

df.columns=['Variance','Skewness','Kurtosis','Entropy','Class']

X=df.iloc[:,:-1]
Y=df.iloc[:,4]
from sklearn import model_selection
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train,Y_train)

Y_train_pred=dtree.predict(X_train)
Y_test_pred=dtree.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train,Y_train_pred)

confusion_matrix(Y_test,Y_test_pred)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
dot_data=StringIO()
export_graphviz(dtree,out_file=dot_data,feature_names=df.columns[:-1],class_names=(str)(df.Class.values))
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('Banknote_Gini_Tree.pdf')

dtree_entropy=DecisionTreeClassifier(criterion='entropy')

dtree_entropy.fit(X_train,Y_train)

Y_pred_entropy_train=dtree_entropy.predict(X_train)

Y_pred_entropy_test=dtree_entropy.predict(X_test)

confusion_matrix(Y_train,Y_pred_entropy_train)
confusion_matrix(Y_test,Y_pred_entropy_test)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
dot_data=StringIO()
export_graphviz(dtree_entropy,out_file=dot_data,feature_names=df.columns[:-1],class_names=(str)(np.unique(df.Class.values)))
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('Banknote_Entropy_Tree.pdf')

dtree.score(X_test,Y_test)
dtree_entropy.score(X_test,Y_test)