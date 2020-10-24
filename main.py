import networkx as nx
import matplotlib.pyplot as plt

path = "/Users/khoihd/Documents/workspace/CP-19"


def plot():
    algorithm = "RDIFF"
    topology = "random-network"
    agent = 10
    instance = 1
    topology_file = path + "/" + algorithm + "/scenario/" + topology + "/d" + str(agent) + "/" + str(instance)
    topology_file += "/topology.ns"

    graph = generate_plot(topology_file)
    nx.draw(graph, with_labels=True)
    plt.show()


def generate_plot(topology_file):
    graph = nx.Graph()
    server_id = ""
    with open(topology_file) as f:
        for line in f.readlines():
            if "large" in line:
                line = line.replace("tb-set-hardware $node", "");
                server_id = line[0]
            if line.startswith("set linkClient"):
                line = line.replace("set linkClient", "")
                node = server(line[0], server_id)
                graph.add_edge(node, "ClientPool-" + line[0])
            elif line.startswith("set link"):
                line = line.replace("set link", "")
                node_one = server(line[0], server_id)
                node_two = server(line[1], server_id)
                graph.add_edge(node_one, node_two)
    return graph


def server(node_id, server_id):
    if node_id == server_id:
        return "Server" + server_id
    else:
        return node_id


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plot()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
