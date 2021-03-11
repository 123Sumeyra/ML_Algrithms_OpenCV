import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from KnnOpenCv.KNN.preprocessing import Preprocessor
from KnnOpenCv.KNN.datasets import DatasetLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')
from imutils import paths

dataset_path = "datasets/Yemek"





print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))
pp = Preprocessor(150, 60) #boyut
dl = DatasetLoader(preprocessors=[pp])
(data, labels) = dl.load(imagePaths, verbose=10)
#print(data.shape)
#print(labels.shape)
data = data.reshape((data.shape[0], -1))# bu kodun doğru boyutu bulmasını sağlıyor
#print(data.shape)
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)#random_state=42(kodu hep aynı yerden bölmek)

#print(trainX.shape)
#print(testX.shape)

print("[INFO] evaluating k-NN classifier...")
k_values =range(1,11)
k_accuracy=[]
for i in k_values:
    model = KNeighborsClassifier(n_neighbors=i, metric='minkowski')
    model.fit(trainX, trainY)
    predict = model.predict(testX)
    k_accuracy.append(metrics.accuracy_score(predict,testY))
print("K_accuracy: ", (k_accuracy))


plt.plot(k_values,k_accuracy, color = 'blue',linestyle='dashed',
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Accuracy  and  K Values')
plt.xlabel('K')
plt.ylabel('Accuracy')
optimal_k_value =k_values[k_accuracy.index(max(k_accuracy))]
print("Maximum accuracy:",max(k_accuracy),"at K =",round(optimal_k_value))

#print(model.score(testX, testY))


print(confusion_matrix(testY,predict))
plt.show()


#print(le.classes_)
#print(classification_report(testY, model.predict(testX)))