# global imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 

# function definitions
def get_data(data):
    if data == 'Iris':
        data = load_iris()
    elif data == 'Wine':
        data = load_wine()
    else:
        data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names, index = None)
    df['Type'] = data.target
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state = 42, test_size = 0.2)
    return X_train, X_test, y_train, y_test, df, data.target_names

def get_classifier(classifier):
    if classifier == "SVC":
        c = st.sidebar.slider(label = "Choose value of C", min_value = 0.0001, max_value = 10.0)
        model = SVC(C = c)
    elif classifier == "Random Forest":
        max_depth = st.sidebar.slider(label = "Choose max depth", min_value = 2, max_value = 10)
        n_estimators = st.sidebar.slider(label = "Choose number of estimators", min_value = 1, max_value = 100)
        model = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, random_state = 42)
    else:
        neighbors = st.sidebar.slider(label = "Choose number of neighbors", min_value = 1, max_value = 20)
        model = KNeighborsClassifier(n_neighbors = neighbors)
    return model 

def get_PCA(df):
    pca = PCA(n_components=3)
    result = pca.fit_transform(df.loc[:, df.columns != 'Type'])
    df['pca-1'] = result[:, 0]
    df['pca-2'] = result[:, 1]
    df['pca-3'] = result[:, 2]
    return df

# streamlit app
st.title("Classifier App in Action")
st.write("This is a simple classifier app that allows you to choose a dataset and a classifier to train the model and visualize the results.")
st.sidebar.title("Classifier Parameters")

# user input for dataset and classifier
dataset = st.selectbox("Which dataset do you want to choose?", options = ['Iris', 'Wine', 'Breast Cancer'])
X_train, X_test, y_train, y_test, df, classes = get_data(dataset)

classifier = st.selectbox("Which classifier do you want to choose?", options = ['KNN', 'SVC', 'Random Forest'])
model = get_classifier(classifier)

st.dataframe(df.sample(n=5, random_state = 42))
st.subheader("Classes")
for idx, val in enumerate(classes):
    st.text("{}:{}".format(idx, val))

# train model 
st.subheader("Model Training")
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
st.text(f"Train score: {round(train_score, 2)}")
st.text(f"Test score: {round(test_score, 2)}")

# visualize pca
st.subheader("PCA Visualization")
df = get_PCA(df)

fig = plt.figure(figsize = (6, 12))
plt.subplots_adjust(right=None)
ax = fig.add_subplot(2, 1, 1)
sns.scatterplot(data = df, x = 'pca-1', y = 'pca-2', hue = 'Type', palette = sns.color_palette("husl", len(classes)), legend = 'full')
ax = fig.add_subplot(2, 1, 2, projection = '3d')
ax.scatter(data = df[df['Type'] == 0], xs = 'pca-1', ys = 'pca-2', zs = 'pca-3', label = '0')
ax.scatter(data = df[df['Type'] == 1], xs = 'pca-1', ys = 'pca-2', zs = 'pca-3', label = '1')
ax.scatter(data = df[df['Type'] == 2], xs = 'pca-1', ys = 'pca-2', zs = 'pca-3', label = '2')
ax.legend()
ax.set_xlabel('pca-1')
ax.set_ylabel('pca-2')
ax.set_zlabel('pca-3')
st.pyplot(fig)