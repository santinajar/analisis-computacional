from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
"sofi estuvo aca"
"Cambio para pull request"
modelo = BayesianNetwork([("C", "A"), ("U", "A")])

cpd_c = TabularCPD(variable="C", variable_card=3, values=[[0.33], [0.33], [0.33]])
cpd_u = TabularCPD(variable="U", variable_card=3, values=[[0.33], [0.33], [0.33]])

cpd_a = TabularCPD(
    variable="A",
    variable_card=3,
    values=[
        [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
        [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
        [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0],
    ],
    evidence=["C", "U"],
    evidence_card=[3, 3],
)

modelo.add_cpds(cpd_c, cpd_u, cpd_a)
modelo.check_model()


from pgmpy.inference import VariableElimination

infer = VariableElimination(modelo)

posterior_p = infer.query(["C"], evidence={"A": 0})
print(posterior_p)