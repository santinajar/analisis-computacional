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

modeloNuevo = BayesianNetwork([("ASIA", "TUB"), ("SMOKE", "LUNG"), ("SMOKE", "BRONC"), ("LUB", "EITHER"), ("LUNG", "EITHER"), ("EITHER", "XRAY"), ("EITHER", "DYSP"), ("BRONC", "DYSP")])
df = pd.read_csv("data_asia.csv")
#print(df)
modeloNuevo.fit(data=df, estimator = MaximumLikelihoodEstimator) 
for i in modeloNuevo.nodes():
    print(modeloNuevo.get_cpds(i)) 
