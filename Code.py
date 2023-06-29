from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

X = [[0, 0], [2, 2], [3, 4], [9, 6], [1, 12]]
y = [1, 0, 1, 0, 1]

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")

valor_sensibilidade = cm[0][0]/(cm[0][0]+cm[0][1])
print(valor_sensibilidade)

valor_especificidade = cm[1][1]/(cm[1][0]+cm[1][1])
print(valor_especificidade)

valor_acuracia = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print(valor_acuracia)

valor_precisao = cm[0][0]/(cm[0][0]+cm[1][0])
print(valor_precisao)

valor_score = 2*((cm[0][0]/cm[0][0]+cm[1][0])*(cm[0][0]/cm[0][0]+cm[0][1])/(cm[0][0]/cm[0][0]+cm[1][0])+(cm[0][0]/cm[0][0]+cm[0][1]))
print(valor_score)


