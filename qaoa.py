import numpy as np
import qutip as qt
import networkx as nx
from scipy.optimize import minimize

class QAOA:
    def __init__(self, graph, p):
        self.graph = graph
        self.p = p
        self.n = graph.number_of_nodes()
        self.hamiltonian = self._create_problem_hamiltonian()
        self.mixer = self._create_mixer_hamiltonian()

    def _create_problem_hamiltonian(self):
        H = qt.Qobj(np.zeros((2**self.n, 2**self.n)))
        for edge in self.graph.edges():
            i, j = edge
            sigma_z_i = qt.sigmaz() if i == 0 else qt.qeye(2)
            sigma_z_j = qt.sigmaz() if j == 0 else qt.qeye(2)
            for k in range(1, self.n):
                if k == i:
                    sigma_z_i = qt.tensor(sigma_z_i, qt.sigmaz())
                elif k == j:
                    sigma_z_j = qt.tensor(sigma_z_j, qt.sigmaz())
                else:
                    sigma_z_i = qt.tensor(sigma_z_i, qt.qeye(2))
                    sigma_z_j = qt.tensor(sigma_z_j, qt.qeye(2))
            H += (qt.qeye(2**self.n) - sigma_z_i * sigma_z_j) / 2
        return H

    def _create_mixer_hamiltonian(self):
        X = qt.sigmax()
        for _ in range(self.n - 1):
            X = qt.tensor(X, qt.sigmax())
        return X

    def _evolve_state(self, initial_state, hamiltonian, t):
        return (-1j * hamiltonian * t).expm() * initial_state

    def cost_function(self, params):
        gamma = params[:self.p]
        beta = params[self.p:]
        
        psi = qt.basis([2]*self.n, [0]*self.n)
        psi = qt.hadamard_transform(N=self.n) * psi

        for i in range(self.p):
            psi = self._evolve_state(psi, self.hamiltonian, gamma[i])
            psi = self._evolve_state(psi, self.mixer, beta[i])

        return -np.real((psi.dag() * self.hamiltonian * psi)[0, 0])

    def optimize(self):
        initial_params = np.random.rand(2 * self.p)
        result = minimize(self.cost_function, initial_params, method='COBYLA')
        return result.x, -result.fun

    def get_solution(self, optimal_params):
        gamma = optimal_params[:self.p]
        beta = optimal_params[self.p:]
        
        psi = qt.basis([2]*self.n, [0]*self.n)
        psi = qt.hadamard_transform(N=self.n) * psi

        for i in range(self.p):
            psi = self._evolve_state(psi, self.hamiltonian, gamma[i])
            psi = self._evolve_state(psi, self.mixer, beta[i])

        probabilities = np.real(psi.diag())
        most_probable_state = np.binary_repr(np.argmax(probabilities), width=self.n)
        return [int(bit) for bit in most_probable_state]