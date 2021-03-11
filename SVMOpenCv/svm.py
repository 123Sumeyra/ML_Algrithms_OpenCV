from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from SVMOpenCv.SVM.preprocessing import Preprocessor
from SVMOpenCv.SVM.datasets import DatasetLoader
from sklearn import svm, metrics
import warnings
warnings.filterwarnings('ignore')
from imutils import paths


dataset_path = "datasets/Yemek"
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))


pp = Preprocessor(150, 60)
dl = DatasetLoader(preprocessors=[pp])
(data, labels) = dl.load(imagePaths, verbose=10)
#print(data.shape)
#print(labels.shape)
data = data.reshape((data.shape[0], -1))# bu kodun doğru boyutu bulmasını sağlıyor
#print(data.shape)
# Import train_test_split function


le = LabelEncoder()
labels = le.fit_transform(labels)
# Split dataset into training set and test set       # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

#print(trainX.shape)
#print(testX.shape)

clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#print("Accuracy",accuracy_score(y_pred,y_test))
print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))
