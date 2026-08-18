# import numpy as np
# import pdb_voxelizier
# import cnn_mlp_encoder
# import jw_quantum_mapper
# from scipy.optimize import minimize
# from scipy.spatial.distance import squareform
# from scipy.linalg import eigh
# import sympy
# import openfermion as op
# import torch
import qiskit
import qiskit_algorithms
import qiskit_aer as q_aer
import qiskit_nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import SLSQP
# Use Qiskit Nature to generate operator pools
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import SlaterDeterminant
from qiskit.quantum_info import SparsePauliOp

# Method that runs the full VQE. Requires existing parameters
def run_vqe(qubit_operators:SparsePauliOp,actual_ground_state:float,ansatz:qiskit.QuantumCircuit,optimizer:qiskit_algorithms.optimizers.Optimizer,estimator:AerEstimator,seed:int=170,callback=None):
    GSE = actual_ground_state
    SYSTEM = qubit_operators
    
    algorithm_globals.random_seed(seed)
    
    vqe = VQE(estimator=estimator,ansatz=ansatz,optimizer=optimizer,callback=callback)
    
    result = vqe.compute_minimum_eigenvalue(operator=SYSTEM)

    criterion = torch.nn.MSELoss()
    
    loss = criterion(result.eigenvalue,GSE)
    
    print(f"Calculated result had a loss of {loss.item()}")
    return

def run_vqe_hea():
    
    pass