from IPython.display import Image
%matplotlib inline

# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

import pandas as pd

wb = gc.open_by_url('https://docs.google.com/spreadsheets/d/1DwM-ul3deq8HRMRq66kPrlCLjaKfycThcCn32zIO45U/edit#gid=0')

sheet = wb.worksheet('Sheet1')

data = sheet.get_all_values()

df = pd.DataFrame(data)
df.columns = df.iloc[0]
df = df.iloc[1:]
df

from sklearn import datasets
import numpy as np

X = df.iloc[:,[4,5]]
y = df.iloc[:,[2]]
print('Nilai', np.unique(y))

X

if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

**MODEL1 PERTAMA PERCEPTRON**

from sklearn.linear_model import Perceptron

mapping = {'A atau AB': 1,
           'B atau BC': 2,
           'C atau D' : 3,
           'E atau T' : 4}

y_train = np.array(y_train).flatten()
y_test = np.array(y_test).flatten()

y_train_encoded = [mapping[val] for val in y_train]
y_test_encoded = [mapping[val] for val in y_test]

ppn = Perceptron(eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train_encoded)

y_pred = ppn.predict(X_test_std)
print(f'Hasil Prediksi y_pred :\n{y_pred}')
print('Hasil Prediksi yang salah: %d\n' % (y_test_encoded != y_pred).sum())

tingkat_kesibukan = X_test_std[:,0]
minat_matakuliah = X_test_std[:,1]

print(f'Ini merupakan Hasil Perbandingan Lama Belajar, Minat, dan Hasil Prediksi :')
for (kesibukan, minat), hasil in zip(zip(tingkat_kesibukan, minat_matakuliah), y_pred):
    print(f'Kesibukan : {kesibukan}, Minat : {minat}, Hasil : {hasil}')

print("\n")
print(f'Sedangkan, ini merupakan Hasil Perbandingan Lama Belajar, Minat, dan Hasil yang seharusnya')
for (kesibukan, minat), hasil in zip(zip(tingkat_kesibukan, minat_matakuliah), y_test):
    print(f'Kesibukan : {kesibukan}, Minat : {minat}, Hasil : {hasil}')


from sklearn.metrics import accuracy_score

print('Hasil Akurasinya : %.2f' % accuracy_score(y_test_encoded, y_pred))

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

ppn.fit(X_train_std,y_train_encoded)
X_combined = np.vstack((X_train_std.astype(float), X_test_std.astype(float)))
y_combined = np.hstack((y_train_encoded, y_test_encoded))

plot_decision_regions(X=X_combined, y=y_combined,
                      clf=ppn, legend=2)
plt.xlabel('Tingkat Kesibukan')
plt.ylabel('Minat Ketertarikan')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

**MODEL2 KEDUA LOGISTIC REGRESSION**

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train_encoded)

plot_decision_regions(X_combined, y_combined,
                      clf=lr, legend=2)
plt.xlabel('Kesibukan')
plt.ylabel('Minat Ketertarikan')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

y_pred = lr.predict(X_test.astype(int))
print('Hasil Prediksi yang Salah : %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

if Version(sklearn_version) < '0.17':
    lr.predict_proba(X_test_std[0, :])
else:
    lr.predict_proba(X_test_std[0, :].reshape(1, -1))

**MODEL3 KETIGA SUPPORT VECTOR MACHINE**

from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train_encoded)

plot_decision_regions(X_combined, y_combined,
                      clf=svm, legend=2)
plt.xlabel('Kesibukan]')
plt.ylabel('Minat Ketertarikan')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
y_pred = svm.predict(X_test.astype(int))
print('Hasil Prediksi yang Salah : %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

#Machine Learning Model --> SVM RBF gamma 10
#Standardized Data
from sklearn.svm import SVC

svm_rbf = SVC(kernel='rbf', random_state=0, gamma=10, C=1.0)
svm_rbf.fit(X_train_std, y_train_encoded)

y_pred = svm_rbf.predict(X_test.astype(int))
print('Misclassified samples: %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

#Machine Learning Model --> SVM RBF gamma 10
#Non-Standardized Data
from sklearn.svm import SVC

svm_rbf = SVC(kernel='rbf', random_state=0, gamma=10, C=1.0)
svm_rbf.fit(X_train, y_train_encoded)

y_pred = svm_rbf.predict(X_test.astype(int))
print('Misclassified samples: %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

#Machine Learning Model --> Tree Depth 3
#Standardized Data
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train_std, y_train_encoded)

y_pred = tree.predict(X_test_std)
print('Misclassified samples: %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

#Machine Learning Model --> Tree Depth 3
#Standardized Data
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)
tree.fit(X_train_std, y_train_encoded)

y_pred = tree.predict(X_test_std)
print('Misclassified samples: %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

from sklearn.tree import export_graphviz

export_graphviz(tree,
                out_file='tree.dot',
                feature_names=['Tingkat kesibukan di luar kuliah',
                               'Minat Keterkaitan'])

import pydotplus

from IPython.display import Image
from IPython.display import display

if Version(sklearn_version) >= '0.18':

    try:

        import pydotplus

        dot_data = export_graphviz(
        tree,
        out_file=None,
        # the parameters below are new in sklearn 0.18
        feature_names=['Tingkat kesibukan di luar kuliah',
                       'Minat Keterkaitan'],
        class_names=['1', '2', '3', '4'],
        filled=True,
        rounded=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        display(Image(graph.create_png()))

    except ImportError:
        print('pydotplus is not installed.')

#Machine Learning Model --> Forest n_estimator 10
#Standardized Data
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train_std, y_train_encoded)

y_pred = forest.predict(X_test_std)
print('Misclassified samples: %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

#Machine Learning Model --> Forest n_estimator 100
#Standardized Data
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=100,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train_std, y_train_encoded)

y_pred = forest.predict(X_test_std)
print('Misclassified samples: %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

#Machine Learning Model --> KNN n=5
#Standardized Data
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train_encoded)

y_pred = knn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

#Machine Learning Model --> KNN n=15
#Standardized Data
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15, p=2, metric='minkowski')
knn.fit(X_train_std, y_train_encoded)

y_pred = knn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

#Machine Learning Model --> KNN n=15
#Standardized Data
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=30, p=2, metric='minkowski')
knn.fit(X_train_std, y_train_encoded)

y_pred = knn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test_encoded != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Hasil Akurasi : %.2f' % accuracy_score(y_test_encoded, y_pred))

x = df[['Jumlah SKS','Jenis kelamin mahasiswa','Tingkat kesibukan di luar kuliah, skala 1 sd 5 (1 sangat tidak sibuk, 5 sangat sibuk)','Minat/ketertarikan pada mata kuliah ini, skala 1 sd 5 (1 sangat tidak tertarik, 5 sangat tertarik)','Minat pemahaman materi kuliah ini, skala 1 sd 5 (1 sangat tidak paham, 5 sangat paham)','Komponen Nilai','Pelaksanaan kuliah','Lama belajar per minggu', 'Total SKS yang diambil saat mengambil kuliah ini', 'Style mengajar dosen, skala 1 sd 5 (1 sangat tidak oke, 5 sangat oke)', 'Style belajar mahasiswa, skala 1 sd 5 (1 sangat males, 5 sangat semangat)', 'Tingkat Kehadiran, skala 1 sd 3 (1 rendah, 2 sedang, 3 tinggi)']].values
y = df[['Nilai']]

x_tree = np.array(x)

y_tree = np.array(y)
y_tree = np.array(y_tree).flatten()
y_tree

if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_tree, y_tree, test_size=0.3, random_state=0)

# Create a mapping of gender categories to numerical values
gender_mapping = {
      'Laki-laki':0,
      'Perempuan':1
}

# Convert gender values in the training and test datasets to numerical values using the mapping
x_train_gender_numeric = [gender_mapping[gender] for gender in x_train[:, [1]].flatten()]
x_test_gender_numeric = [gender_mapping[gender] for gender in x_test[:, [1]].flatten()]

component_value_mapping = {
    'Ada Kuis, Ada Tubes, Ada UTS':1,
    'Ada Kuis, Ada Tubes, Ada UTS, Ada UAS':2,
    'Ada Kuis, Ada UTS, Ada UAS':3,
    'Ada PR, Ada Kuis, Ada Tubes, Ada UTS':4,
    'Ada PR, Ada Kuis, Ada Tubes, Ada UTS, Ada UAS':5,
    'Ada PR, Ada Kuis, Ada UTS, Ada UAS':6,
    'Ada PR, Ada Tubes':7,
    'Ada PR, Ada Tubes, Ada UTS':8,
    'Ada PR, Ada Tubes, Ada UTS, Ada UAS':9,
    'Ada PR, Ada UAS':10,
    'Ada PR, Ada UTS, Ada UAS':11,
    'Ada Tubes':12,
    'Ada Tubes, Ada UTS':13,
    'Ada Tubes, Ada UTS, Ada UAS':14,
    'Ada UTS, Ada UAS':15
}

x_train_component_values = [component_value_mapping[component] for component in x_train[:, [5]].flatten()]
x_test_component_values = [component_value_mapping[component] for component in x_test[:, [5]].flatten()]

learning_system_mapping = {
    'Daring asinkron (video saja)': 1,
    'Daring sinkron':2,
    'Daring sinkron, Daring asinkron (video saja)':3,
    'Luring':4,
    'Luring, Daring asinkron (video saja)':5,
    'Luring, Daring sinkron':6,
    'Luring, Daring sinkron, Daring asinkron (video saja)':7
}

x_train_learning_system = [learning_system_mapping[code] for code in x_train[:, [6]].flatten()]
x_test_learning_system = [learning_system_mapping[code] for code in x_test[:, [6]].flatten()]

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(x_train, y_train)

y_pred = tree.predict(x_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz

export_graphviz(tree,
                out_file='tree.dot',
                feature_names=['Jumlah SKS',
                 'Jenis kelamin mahasiswa',
                 'Tingkat kesibukan di luar kuliah, skala 1 sd 5 (1 sangat tidak sibuk, 5 sangat sibuk)',
                 'Minat/ketertarikan pada mata kuliah ini, skala 1 sd 5 (1 sangat tidak tertarik, 5 sangat tertarik)',
                 'Minat pemahaman materi kuliah ini, skala 1 sd 5 (1 sangat tidak paham, 5 sangat paham)',
                 'Komponen Nilai',
                 'Pelaksanaan kuliah',
                 'Lama belajar per minggu',
                 'Total SKS yang diambil saat mengambil kuliah ini',
                 'Style mengajar dosen, skala 1 sd 5 (1 sangat tidak oke, 5 sangat oke)',
                 'Style belajar mahasiswa, skala 1 sd 5 (1 sangat males, 5 sangat semangat)',
                 'Tingkat Kehadiran, skala 1 sd 3 (1 rendah, 2 sedang, 3 tinggi)'])

import pydotplus

from IPython.display import Image
from IPython.display import display

if Version(sklearn_version) >= '0.18':

    try:

        import pydotplus

        dot_data = export_graphviz(
            tree,
            out_file=None,
            # the parameters below are new in sklearn 0.18
            feature_names=['Jumlah SKS',
                 'Jenis kelamin mahasiswa',
                 'Tingkat kesibukan di luar kuliah, skala 1 sd 5 (1 sangat tidak sibuk, 5 sangat sibuk)',
                 'Minat/ketertarikan pada mata kuliah ini, skala 1 sd 5 (1 sangat tidak tertarik, 5 sangat tertarik)',
                 'Minat pemahaman materi kuliah ini, skala 1 sd 5 (1 sangat tidak paham, 5 sangat paham)',
                 'Komponen Nilai',
                 'Pelaksanaan kuliah',
                 'Lama belajar per minggu',
                 'Total SKS yang diambil saat mengambil kuliah ini',
                 'Style mengajar dosen, skala 1 sd 5 (1 sangat tidak oke, 5 sangat oke)',
                 'Style belajar mahasiswa, skala 1 sd 5 (1 sangat males, 5 sangat semangat)',
                 'Tingkat Kehadiran, skala 1 sd 3 (1 rendah, 2 sedang, 3 tinggi)'],
            class_names=['1', '2', '3', '4'],
            filled=True,
            rounded=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        display(Image(graph.create_png()))

    except ImportError:
        print('pydotplus is not installed.')


y_combined.size