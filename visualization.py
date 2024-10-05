import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_graph_solution(graph, solution):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color=solution, cmap=plt.cm.RdYlBu)
    nx.draw_networkx_labels(graph, pos)
    plt.axis('off')
    plt.title("QAOA Solution for MaxCut")
    plt.show()

def plot_energy_landscape(qaoa, resolution=20):
    gamma = np.linspace(0, np.pi, resolution)
    beta = np.linspace(0, np.pi, resolution)
    energy_landscape = np.zeros((resolution, resolution))

    for i, g in enumerate(gamma):
        for j, b in enumerate(beta):
            energy_landscape[i, j] = -qaoa.cost_function([g, b])

    plt.figure(figsize=(10, 8))
    plt.imshow(energy_landscape, extent=[0, np.pi, 0, np.pi], origin='lower', cmap='viridis')
    plt.colorbar(label='Energy')
    plt.xlabel('Beta')
    plt.ylabel('Gamma')
    plt.title('QAOA Energy Landscape')
    plt.show()

def plot_convergence(optimization_result):
    plt.figure(figsize=(10, 6))
    plt.plot(optimization_result.nfev, -optimization_result.fun, 'b-')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Max Cut Value')
    plt.title('QAOA Convergence')
    plt.grid(True)
    plt.show()