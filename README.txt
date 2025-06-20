MODRED repository


Reduction methods :
MODAL_REDUCTION.M : modal reduction
POD_REDUCTION.M : Proper Orthogonal Decomposition reduction
BALANCED_MODEL_REDUCTION.M : classical balanced reduction
BALANCED_POD_REDUCTION_EUL.M : balanced reduction for large systems with Euler solver
BALANCED_POC_REDUCTION_ODE.M : balanced reduction for large systems with ODE solver
SUBSPACE_IDENTIFICATION.M : Subspace identification
SUBSPACE_IDENTIFICATION_LARGE_SCALE.M : Subspace identification for large systems
NARMAX_SISO.M : non linear ARX model identification


Documentation :
REDUCTION DE MODELES.PDF

A test code is proposed with :

COMPARAISON_REDUCTION_METHODS.M that is related to model reduction for heat conduction in a wall model coded in TUTORIAL.M
TEST_NARMAX.M that evaluate NARMAX system identification with optimal choice for model orders

Additional functions :
D2C_ZOH.M : discrete to continuous model
LANCZOS_ARNOLDI.M : algorithms of Lanczos and Arnoldi for eigenvalues calculation
TF2SS_MIMO.M : transfer function to state space model with MIMO systems
SIMULATION_REDUCED_MODEL.M : time simulation using ODE solvers
CUSTOM_LYAP.M : solve the Lyapunov equations
CUSTOM_KRON.M : make Kronoker matrices
MY_SS.M : create a SYS state-space model
COMPUTE_RMSE : compute the square root of the norm2 of the error

Jean-Luc Battaglia
University of Bordeaux
