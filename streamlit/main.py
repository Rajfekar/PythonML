import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


datasetName = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Diabetes"))
clfName = st.sidebar.selectbox("Select Classifier",("DecitionTreeClassifier" ,"Random Forest","KNN","SVC"))
st.title(clfName)
def getDataset(datasetName):
    if(datasetName == "Iris"):
        data = datasets.load_iris()
    elif(datasetName == "Breast Cancer"):
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_diabetes()
    X = data.data
    y = data.target
    return X,y


X, y = getDataset(datasetName)
st.write("Shape of the dataset: " , X.shape)
st.write("Number of classes: ", len(np.unique(y)))

def addParameterUi(clfName):
    params = dict()
    if(clfName == "KNN"):
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif(clfName == "DecitionTreeClassifier"):
         criterion = st.sidebar.selectbox("criterion",("gini","entropy","log_loss"))
         splitter = st.sidebar.selectbox("splitter",("best","random"))
         max_depth = int(st.sidebar.number_input("max_depth",value=4))
         max_leaf_nodes = int(st.sidebar.slider("max_leaf_nodes",2,10))
         min_samples_split = st.sidebar.number_input("min_samples_split",value=2)
         min_samples_leaf = st.sidebar.number_input("min_samples_leaf",value=1)
         min_weight_fraction_leaf = st.sidebar.number_input("min_weight_fraction_leaf")
         ccp_alpha =  st.sidebar.number_input("ccp_alpha",1,100)
         params["criterion"] = criterion
         params["splitter"] = splitter
         params["max_depth"] = max_depth
         params["max_leaf_nodes"] = max_leaf_nodes
         params["min_samples_split"] = min_samples_split
         params["min_samples_leaf"] = min_samples_leaf
         params["min_weight_fraction_leaf"] = min_weight_fraction_leaf
         params["ccp_alpha"] = ccp_alpha
    elif(clfName == "SVM"):
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    else:
        maxDepth = st.sidebar.slider("maxDepth",2,15)
        noEstimators = st.sidebar.slider("noEstimators",1,100)
        params["maxDepth"] = maxDepth
        params["noEstimators"] = noEstimators
    return params

params = addParameterUi(clfName)

def getClassifier(clfName,params):

    if(clfName == "KNN"):
            clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif(clfName == "DecitionTreeClassifier"):
         clf= DecisionTreeClassifier(criterion=params["criterion"],
                                     splitter=params["splitter"],
                                     max_depth=params["max_depth"],
                                     max_leaf_nodes=params["max_leaf_nodes"],
                                     min_samples_split=params["min_samples_split"],
                                     min_samples_leaf=params["min_samples_leaf"],
                                     min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
                                     ccp_alpha=params["ccp_alpha"]
                                     )
    elif(clfName == "SVM"):
            clf = SVC(C=params["C"])
    else:
            clf = RandomForestClassifier(n_estimators=params["noEstimators"],max_depth=params["maxDepth"],random_state=1234)
    return clf

clf = getClassifier(clfName,params)
# Classification 
x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
acc_score = round(accuracy_score(y_test,y_pred),2)
st.write(f"classifier = {clfName}")
st.write(f"accuracy = {acc_score}")

# Plot
pca = PCA(2)
# unsupervised not required label
X_projected = pca.fit_transform(X)
x1 = X_projected[:,0]
x2 = X_projected[:,1]
fig = plt.figure(figsize=(10,10))
plt.scatter(x1, x2, c=y, alpha=.8,cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.pyplot(fig)



