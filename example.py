import networkx as nx
from qaoa import QAOA
from visualization import plot_graph_solution, plot_energy_landscape, plot_convergence

def main():
    # Create a sample graph
    G = nx.random_regular_graph(3, 10)

    # Initialize QAOA
    qaoa = QAOA(G, p=2)

    # Optimize QAOA parameters
    optimal_params, max_cut_value = qaoa.optimize()

    # Get the solution
    solution = qaoa.get_solution(optimal_params)

    # Visualize results
    plot_graph_solution(G, solution)
    plot_energy_landscape(qaoa)
    
    # Print results
    print(f"Optimal parameters: {optimal_params}")
    print(f"Max Cut value: {max_cut_value}")
    print(f"Solution: {solution}")

if __name__ == "__main__":
    main()