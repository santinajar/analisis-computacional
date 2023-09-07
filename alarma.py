from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy . sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd

modelo = BayesianNetwork([("R", "A"), ("S", "A"), ("A", "J"), ("A", "M")])

cpd_r = TabularCPD(variable="R", variable_card=2, values=[[0.01], [0.99]])
cpd_s = TabularCPD(variable="S", variable_card=2, values=[[0.02], [0.98]])

cpd_a = TabularCPD(
    variable="A",
    variable_card=2,
    values=[
        [0.95, 0.94 , 0.29, 0.001],
        [0.05, 0.06, 0.71 , 0.999],
    ],
    evidence=["R", "S"],
    evidence_card=[2, 2],
)

cpd_j = TabularCPD(
    variable="J",
    variable_card=2,
    values=[
        [0.9, 0.01 ],#PREGUNTA 11 [[0.9, 0.05 ],[0.1, 0.95],]
        [0.1, 0.99],
    ],
    evidence=["A"],
    evidence_card=[2],
)

cpd_m = TabularCPD(
    variable="M",
    variable_card=2,
    values=[
        [0.7, 0.01],
        [0.3, 0.99],
    ],
    evidence=["A"],
    evidence_card=[2],
)
modelo.add_cpds(cpd_r, cpd_s, cpd_a, cpd_j, cpd_m)
modelo.check_model()

from pgmpy.inference import VariableElimination

infer = VariableElimination(modelo)
posterior_p = infer.query(["R"], evidence={"M": 1,"J":1})
#print(posterior_p)




#taller 4

mod = BayesianNetwork([("R", "A"), ("S", "A"), ("A", "J"), ("A", "M")])

samples = BayesianModelSampling(modelo).forward_sample(size=int(1e5))
#print(samples.head(n=50))

emv = MaximumLikelihoodEstimator(model=mod, data=samples)

# Estimar para nodos sin padres
cpdem_r = emv.estimate_cpd(node="R")
#print(cpdem_r)
cpdem_s = emv.estimate_cpd(node="S")
#print(cpdem_s)

# Estimar para nodo Alarma
cpdem_a = emv.estimate_cpd(node="A")
#print(cpdem_a)


mod.fit(data=samples, estimator = MaximumLikelihoodEstimator) 
for i in mod.nodes():
    #print(mod.get_cpds(i)) 
    x=0

from pgmpy.estimators import BayesianEstimator

eby = BayesianEstimator(model=mod, data=samples)

cpdby_r = eby.estimate_cpd(node="R", prior_type="dirichlet", pseudo_counts=[[2000000], [200000]])
#print(cpdby_r)

#punto 3

# Define el modelo de red bayesiana
modeloNuevo = BayesianNetwork([
    ("asia", "tub"),
    ("smoke", "lung"),
    ("smoke", "bronc"),
    ("lung", "either"),
    ("bronc", "either"),
    ("either", "xray"),
    ("either", "dysp"),
    ("bronc", "dysp")
])

# Carga los datos desde el archivo CSV
df = pd.read_csv("data_asia.csv")


# Elimina la columna "Unnamed: 0" si es necesario
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Ajusta el modelo utilizando MaximumLikelihoodEstimator
modeloNuevo.fit(df, estimator=MaximumLikelihoodEstimator)
for i in modeloNuevo.nodes():
    print(modeloNuevo.get_cpds(i)) 
"""""
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test = train_test_split(df, test_size=0.30, random_state=(42))
#print(df)
modeloNuevo.fit(data=X_train, estimator = MaximumLikelihoodEstimator) 
for i in modeloNuevo.nodes():
    print(modeloNuevo.get_cpds(i)) 
modeloNuevo.fit(data=X_test, estimator = MaximumLikelihoodEstimator) 
for i in modeloNuevo.nodes():
    print(modeloNuevo.get_cpds(i)) 

y_true = [...]  # Etiquetas reales de los datos de prueba
y_pred = [...]  # Predicciones de tu modelo para los datos de prueba    
exactitud = accuracy_score(y_true, y_pred)
    

y_true = [...]  # Etiquetas reales de los datos de prueba
y_pred = [...]  # Predicciones de tu modelo para los datos de prueba

# Calcular la matriz de confusión
confusion = confusion_matrix(y_true, y_pred)

# Extraer los valores TP, FP, TN y FN de la matriz de confusión
TP = confusion[1, 1]  # Verdaderos Positivos
FP = confusion[0, 1]  # Falsos Positivos
TN = confusion[0, 0]  # Verdaderos Negativos
FN = confusion[1, 0]  # Falsos Negativos

"""""
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy . sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

x_train, x_test, y_train, y_test = train_test_split(df.drop('lung', axis=1), df['lung'], test_size=0.30, random_state=50)
#print(df)
#modeloNuevo.fit(data=x_train, estimator = MaximumLikelihoodEstimator) 
#for i in modeloNuevo.nodes():
    #print(modeloNuevo.get_cpds(i)) 
#modeloNuevo.fit(data=x_test, estimator = MaximumLikelihoodEstimator) 
#for i in modeloNuevo.nodes():
   # print(modeloNuevo.get_cpds(i)) 
X = df.drop('lung', axis=1)
y = df['lung']
y_pred = modeloNuevo.predict(x_test)  # Predicciones de tu modelo para los datos de prueba    
exactitud = accuracy_score(y_test, y_pred)
    

matriz_confusion = confusion_matrix(y_test, y_pred)

tp = matriz_confusion[1, 1] #verdaderos positivos
fp = matriz_confusion[0, 1]  #falsos positivos
tn = matriz_confusion[0, 0]  #verdaderos negarivos
fn = matriz_confusion[1, 0]  #flasos negarivos

# Imprimir los resultados
print(exactitud)
print(tp)
print(fp)
print(tn)
print(fn)

