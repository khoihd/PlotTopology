import os
from glob import glob
from os import listdir, walk
from os.path import isfile, join
import Request
import re
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import plotly.figure_factory as ff
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
from graphviz import Graph

SUCCESS = True
FAILURE = False

path = "/Users/khoihd/Documents/workspace/CP-19"
yachay_path = "khoi.hoang@yachay.seas.wustl.edu:/home/research/khoi.hoang/comparison_new_config"
server_color = "red"
client_color = "green"
node_color = "grey"
client_edge = 1.5
node_edge = 1


def get_demand(algorithm, instance_path_demand):
    # instance_path_demand = path + "/" + algorithm + "/scenario/" + topology + "/d" + str(agent) + "/" + str(instance)
    output_folder = instance_path_demand + "/output"
    zip_file = instance_path_demand + "/" + algorithm + "_output.zip"
    log_file = output_folder + "/map.log"

    if os.path.exists(zip_file) and not os.path.exists(output_folder):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(instance_path_demand)

    # {
    # AppCoordinates {com.bbn, test-service3, 1}={},
    # AppCoordinates {com.bbn, test-service2, 1}={
    #     C={NodeAttribute {QueueLength, true}=72.70782282928928, NodeAttribute {TASK_CONTAINERS, false}=64.09842510775468},
    #     E={NodeAttribute {QueueLength, true}=70.88912809379809, NodeAttribute {TASK_CONTAINERS, false}=62.49508362728623},
    #     G={NodeAttribute {QueueLength, true}=69.30142306065248, NodeAttribute {TASK_CONTAINERS, false}=61.095380153847984}},
    #
    # AppCoordinates {com.bbn, test-service1, 1}={
    #     C={NodeAttribute {QueueLength, true}=120.06186240017193, NodeAttribute {TASK_CONTAINERS, false}=78.20458944733167},
    #     E={NodeAttribute {QueueLength, true}=119.30129391260914, NodeAttribute {TASK_CONTAINERS, false}=77.7091786222174},
    #     G={NodeAttribute {QueueLength, true}=111.62371177910762, NodeAttribute {TASK_CONTAINERS, false}=72.70823871760872}}
    # }

    total_demand = dict()
    # df = pd.DataFrame()
    print(log_file)
    with open(log_file) as f:
        for line in f.readlines():
            if 'Server Demand Inferred' in line:
                run_line = re.search("DCOP Run [0-9]+", line)
                run = run_line.group(0).replace("DCOP Run ", "")
                # print("run =", run)

                line = line.split('Server Demand Inferred ')[1]
                # print(line)

                # server_demand_regex = re.findall("AppCoordinates \{com.bbn, test-service[0-9], 1\}=\{[A-Z]+=\{NodeAttribute \{QueueLength, true\}=([0-9]+\.[0-9]+), NodeAttribute \{TASK_CONTAINERS, false\}=([0-9]+\.[0-9]+)", line)
                server_demand_matches = re.findall("[A-Z]+=\{NodeAttribute \{QueueLength, true\}=[0-9]+\.[0-9]+, NodeAttribute \{TASK_CONTAINERS, false\}=[0-9]+\.[0-9]+", line)
                for client_demand in server_demand_matches:
                    client = client_demand[0]
                    demand = re.search("\{TASK_CONTAINERS, false\}=[0-9]+.[0-9]+", client_demand).group(0).replace("{TASK_CONTAINERS, false}=", "")
                    # print(client, demand)
                    update_total_demand(total_demand, run, "ClientPool-" + client, demand)
        # for run, demand in total_demand.items():
        #     # print(run)
        #     for region, value in demand.items():
        #         # print(region, "= ", end="")
        #         # print("{:.2f}".format(value))
        #         df[run][region] = round(value, 2)
        #         print(df)
    df = pd.DataFrame(total_demand)
    df.fillna(0, inplace=True)
    df = df.round(2)

    return df.T


def update_total_demand(total_demand, run, client, demand):
    # run -> client -> demand
    client_demand = dict()
    if run in total_demand:
        client_demand = total_demand[run]

    if client in client_demand:
        client_demand[client] += float(demand)
    else:
        client_demand[client] = float(demand)

    total_demand[run] = client_demand


def getRequestsOverTime(algorithm, instance_path):
    start_time = 30000
    time_between_batches = 60000

    zip_file = instance_path + "/" + algorithm + "_output.zip"
    output_folder = instance_path + "/output"
    if os.path.exists(zip_file) and not os.path.exists(output_folder):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(instance_path)

    client_folder = [x for x in glob(output_folder + "/*/") if "clientPool" in x]

    result_over_time = dict()
    for demand_folder in client_folder:
        # print(demand_folder)
        demand_file = next(walk(demand_folder))[2][0]
        client = demand_file[-6:-5]
        # print(client)
        # print(demand_file)
        with open(demand_folder + "/" + demand_file) as f:
            line = f.readline()
            while line:
                line = line.strip()
                # "startTime" : 30000,
                if "startTime" in line:
                    line = line.replace("\"startTime\" : ", "")
                    line = line.replace(",", "")
                    run = (int(line) - start_time) // time_between_batches + 1
                    # run = int(line)
                if "ncpContacted" in line:
                    line = f.readline() # read the next line
                    line = line.replace(" ", "")
                    line = line.replace("\"", "")
                    region = line.replace("name:node", "")[0]
                elif "numClients" in line:
                    line = line.replace(" ", "")
                    line = line.replace("\"numClients\":", "")
                elif "artifact" in line:
                    line = line.replace(" ", "")
                    line = line.replace("\"artifact\":\"test-service", "")
                    service = line.replace(",", "")
                elif "serverResult" in line:
                    if "SUCCESS" in line or "SLOW" in line:
                        increment_over_time(result_over_time, region, run, SUCCESS)
                    elif "FAIL" in line:
                        increment_over_time(result_over_time, region, run, FAILURE)
                line = f.readline()

    total_success = 0
    total_fail = 0
    success = list()
    region_success_fail = dict()
    aggregate_request = Request.Request()
    for run, region_map in result_over_time.items():
        # print("run=", run, ":", sep="")
        success_per_run = 0
        fail_per_run = 0
        for region, request in region_map.items():
            total_success += request.success
            total_fail += request.fail
            # print("region=", region, ", request=", request, sep="")
            success_per_run += request.success
            fail_per_run += request.fail

            aggregate_request.success += request.success
            aggregate_request.fail += request.fail

            total_regions(region_success_fail, region, request)
        # print("{}; ".format(success_per_run), end="")
        success.append(success_per_run)

    print()
    total = total_success + total_fail
    # print('total_success={}; success_rate={}'.format(total_success, total_success / total))
    # print('total_fail={}; failure_rate={}'.format(total_fail, total_fail / total))

    return success, region_success_fail, aggregate_request, result_over_time


def increment_over_time(over_time, region, run, flag):
    # if region != datacenter:
    #     return

    region_map = dict()
    request = Request.Request()

    if run in over_time:
        region_map = over_time[run]

    if region in region_map:
        request = region_map[region]

    if flag == SUCCESS:
        request.increment_success()
    elif flag == FAILURE:
        request.increment_fail()

    region_map[region] = request
    over_time[run] = region_map


def total_regions(over_time, region, req):
    # if region != datacenter:
    #     return

    request = Request.Request()

    if region in over_time:
        request = over_time[region]

    request.success += req.success
    request.fail += req.fail

    over_time[region] = request


def plot():
    algorithm = "RDIFF"
    agent = 10
    (topology, instance) = ("scale-free-tree", 0)
    # (topology, instance) = ("random-network", 1)
    topology_file = path + "/" + algorithm + "/scenario/" + topology + "/d" + str(agent) + "/" + str(instance)
    topology_file += "/topology.ns"

    graph = generate_plot(topology_file)
    graph.view()


def generate_plot(topology_file):
    graph = Graph("topology", filename='topology.gv', engine='neato')
    # graph = Graph("topology", filename='topology.gv', engine='neato', outputorder = 'breadthfirst')
    delay_re = "[0-9]+.0ms"
    server_id = ""
    # Add color to server node
    # Add color to client node
    with open(topology_file) as f:
        for line in f.readlines():
            if "large" in line:
                line = line.replace("tb-set-hardware $node", "")
                server_id = line[0]
            if line.startswith("set linkClient"):
                line = line.replace("set linkClient", "")
                node = server(line[0], server_id)
                client = "ClientPool-" + line[0]

                add_node(graph, node)
                add_node(graph, client)
                add_edge(graph, node, client, client_edge, str(1))
            elif line.startswith("set link"):
                line = line.replace("set link", "")
                node_one = server(line[0], server_id)
                node_two = server(line[1], server_id)

                add_node(graph, node_one)
                add_node(graph, node_two)

                label = re.findall(delay_re, line)[0]
                add_edge(graph, node_one, node_two, node_edge, label)
    print(graph.source)
    graph.save()
    graph.render('topology')
    return graph


def add_node(graph, node):
    if "Server" in node:
        graph.node(node, style='filled', fillcolor=server_color)
    elif "Client" in node:
        graph.node(node, style='filled', fillcolor=client_color)
    else:
        graph.node(node, style='filled', fillcolor=node_color)


def add_edge(graph, node_one, node_two, edge_length, label):
    graph.edge(node_one, node_two, len=str(edge_length), label=label)


def server(node_id, server_id):
    if node_id == server_id:
        return "Server-" + server_id
    else:
        return node_id


def color(node):
    if "Server" in node:
        return server_color
    elif "Client" in node:
        return client_color
    else:
        return node_color


def generate_plot_old(topology_file):
    graph = nx.Graph()
    color_map = list()
    server_id = ""
    # Add color to server node
    # Add color to client node
    with open(topology_file) as f:
        for line in f.readlines():
            if "large" in line:
                line = line.replace("tb-set-hardware $node", "")
                server_id = line[0]
            if line.startswith("set linkClient"):
                line = line.replace("set linkClient", "")
                node = server(line[0], server_id)
                client = "ClientPool-" + line[0]

                add_node(graph, node, color_map)
                add_node(graph, client, color_map)

                graph.add_edge(node, client)
            elif line.startswith("set link"):
                line = line.replace("set link", "")
                node_one = server(line[0], server_id)
                node_two = server(line[1], server_id)

                add_node(graph, node_one, color_map)
                add_node(graph, node_two, color_map)

                graph.add_edge(node_one, node_two)
    return graph, color_map


def add_node_old(graph, node, color_map):
    if node not in graph.nodes:
        graph.add_node(node)
        if "Server" in node:
            color_map.append(server_color)
        elif "Client" in node:
            color_map.append(client_color)
        else:
            color_map.append(node_color)


def read_and_plot_requests():
    # Get successful requests
    rdiff_requests, rdiff_regions, rdiff_total = getRequestsOverTime("RDIFF")
    rc_diff_requests, rc_diff_regions, rc_diff_total = getRequestsOverTime("RC-DIFF")
    # Plot successful requests
    fig, ax = plt.subplots()
    ax.plot(rdiff_requests, label='RDIFF')
    ax.plot(rc_diff_requests, label='RC-DIFF')
    ax.set_xlabel("DCOP run")
    ax.set_ylabel("Number of successful requests")
    ax.legend()
    ax.set_title("Number of Successful Requests out of 600 Requests per DCOP run")
    fig.savefig("Successful_Requests_Threshold=0.1.pdf", bbox_inches='tight')

    print("RDIFF", rdiff_total)
    print(rdiff_regions)

    print("RC-DIFF", rc_diff_total)
    print(rc_diff_regions)


def read_and_plot_demand(alg, threshold, instance_path):
    demand_to_plot = get_demand(alg, instance_path)
    # print(demand_to_plot)
    # To PDF
    size = (np.array(demand_to_plot.shape[::-1]) + np.array([0, 1])) * np.array([3, 2])
    # fig, ax = plt.subplots(figsize=(9, 30))
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    # rdiff_table = ax.table(cellText=rdiff.values, colLabels="ClientPool-" + rdiff.columns, rowLabels=rdiff.index, bbox=[0, 0, 1.75, 0.75], cellLoc='center')
    # rdiff_table = ax.table(cellText=rdiff.values, colLabels="ClientPool-" + rdiff.columns, rowLabels=rdiff.index,
    #                        bbox=[0, 0, 1, 1], cellLoc='center')
    rdiff_table = ax.table(cellText=demand_to_plot.values, colLabels="ClientPool-" + demand_to_plot.columns, rowLabels=demand_to_plot.index, bbox=[0, 0, 1, 1])
    rdiff_table.auto_set_font_size(False)
    rdiff_table.set_fontsize(44)
    # fig.savefig("RDIFF_demand.pdf", bbox_inches='tight', orientation='landscape')
    # fig.savefig("RDIFF_demand.png")

    rdiff_pdf = PdfPages("Demand_" + alg + "_" + threshold + ".pdf")
    rdiff_pdf.savefig(fig, bbox_inches='tight')
    rdiff_pdf.close()
    plt.close(fig)

    #
    # print(rdiff)
    # print(rc_diff)
    
    return demand_to_plot


if __name__ == "__main__plot_topology":
    path = "/Users/khoihd/Documents/workspace/CP-19/"
    (topology, instance) = ("scale-free-tree", 0)
    agent = 20
    instance_path = path + topology + "/d" + str(agent) + "/" + str(instance)
    instance_path += "/topology.ns"
    graph = generate_plot(instance_path)
    graph.view()


def unzip_output(instance_path_unzip, alg):
    zip_file = instance_path_unzip + "/" + alg + "_output.zip"
    output_folder = instance_path_unzip + "/output"
    os.system("rm -rf " + output_folder)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(instance_path_unzip)


def get_all_clients(instance_path):
    clients = set()
    for folder in os.listdir(instance_path + "/output"):
        if "clientPool" in folder:
            clients.add(folder.replace("clientPool", ""))

    return clients


# 2020-12-13/08:14:50.410/-0600 [DCOP-nodeA] INFO com.bbn.map.dcop.AbstractDcopAlgorithm [] {}- DCOP Run 1 Region Plan Region A: {AppCoordinates {com.bbn, test-service1, 1}={A=1.0, D=0.0, G=0.0, I=0.0, J=0.0, M=0.0}}
# 2020-12-13/08:14:50.570/-0600 [DCOP-nodeI] INFO com.bbn.map.dcop.AbstractDcopAlgorithm [] {}- DCOP Run 1 Region Plan Region I: RegionPlan [ region: I timestamp: 0 plan: {} ]
def compute_capacity(instance_path, alg, agent_count):
    log_file = instance_path + "/output/map.log"
    client_set = get_all_clients(instance_path)

    run_dict = dict()
    with open(log_file) as f:
        for line in f.readlines():
            if "Region Plan" in line:
                line = line.split("{}- ")[1] # Remove the previous content
                dcop_run_regex = "DCOP Run [0-9]+"
                region_regex = "Region Plan Region [A-Z]"
                plan_regex = "[A-Z]+=[0-1].[0-9]*"

                dcop_run = int(re.findall(dcop_run_regex, line)[0].replace("DCOP Run ", ""))
                if dcop_run not in run_dict:
                    run_dict[dcop_run] = set()

                region = re.findall(region_regex, line)[0].replace("Region Plan Region ", "")
                if region in client_set:
                    run_dict[dcop_run].add(region)

                if "AppCoordinates" in line:
                    plans = re.findall(plan_regex, line)
                    for plan in plans:
                        neighbor = plan.split("=")[0]
                        ratio = float(plan.split("=")[1])
                        if ratio > 0:
                            run_dict[dcop_run].add(neighbor)
    total_non_zero_region = 0
    max_non_zero_region = 0
    for _, non_zero_region in run_dict.items():
        num_region = len(non_zero_region)
        total_non_zero_region += num_region
        max_non_zero_region = max(max_non_zero_region, num_region)
    
    return total_non_zero_region / len(run_dict), max_non_zero_region


# Compute the aggregate capacities of regions that are involved in the DCOP plan
def compute_alg_capacity():
    # capacity_path = "/Users/khoihd/Downloads/hardcode_comparison_new_config"
    # capacity_path = "/Users/khoihd/Downloads/comparison_new_config"
    capacity_path = "/Users/khoihd/Downloads/MAP_khoi_changes/comparison_new_config"
    algorithms = ["RC-DIFF"]
    agents = [5, 10, 15]
    # (topology, instances) = ("scale-free-tree", range(10))
    (topology, instances) = ("random-network", range(10))
    for agent in agents:
        for alg in algorithms:
            total_avg_cap = 0
            total_max_cap = 0
            for instance in instances:
                instance_path = capacity_path + "/" + alg + "/scenario/" + topology + "/d" + str(agent) + "/" + str(instance)
                print(instance_path)
                unzip_output(instance_path, alg)
                avg_cap, max_cap = compute_capacity(instance_path, alg, agent)
                total_avg_cap += avg_cap
                total_max_cap += max_cap

            # print("Average total algorithm capacity =", total_avg_cap/len(instances) * 20)
            print("Average max algorithm capacity =", total_max_cap/len(instances) * 20)


def get_success_ratio(instance_path):
    simulation_folder = instance_path + "/output/simulation"
    total_success = 0
    for root, folders, files in os.walk(simulation_folder):
        for file in files:
            if "final-state.json" in file:
                with open(root + "/" +  file, 'r') as f:
                    for line in f.readlines():
                        if "numRequestsSucceeded" in line:
                            line = line.replace('"numRequestsSucceeded" : ', '').replace(',', '')
                            line = line.replace(' ', '')
                            total_success += int(line)

    return round(total_success / 18000.0 * 100.0, 2)


# Compute the result to fill up the table
def get_s_f(over_time_result, cutoff_ts):
    total_s = 0
    total_f = 0
    for time_step, region_sf in over_time_result.items():
        if time_step >= cutoff_ts:
            for region, region_rq in region_sf.items():
                total_s += region_rq.success
                total_f += region_rq.fail

    return total_s, total_f


def compute_result_table():
    # table_path = "/Users/khoihd/Downloads/hardcode_toy_test"
    # table_path = "/Users/khoihd/Downloads/MAP/hardcode_comparison_new_config"
    # table_path = "/Users/khoihd/Downloads/MAP_updated_changes/comparison_new_config"
    table_path = "/Users/khoihd/Downloads/MAP_khoi_changes/comparison_new_config"

    algorithms = ["RC-DIFF"]
    agents = [10]
    (topology, instances) = ("scale-free-tree", range(1))
    # (topology, instances) = ("random-network", range(10))

    cutoff_timestep = 16
    for agent in agents:
        for alg in algorithms:
            avg_request = Request.Request()
            for instance in instances:
                instance_path = table_path + "/" + alg + "/scenario/" + topology + "/d" + str(agent) + "/" + str(instance)
                print(instance_path)
                unzip_output(instance_path, alg)
                _, region_result, request_total, over_time_result = getRequestsOverTime(alg, instance_path)

                cutoff_success, cutoff_fail = get_s_f(over_time_result, cutoff_timestep)
                print("success={}, total={}, ratio={}".format(cutoff_success, cutoff_success + cutoff_fail, cutoff_success / (cutoff_success + cutoff_fail) * 100))
                for run, result in over_time_result.items():
                    print(run, result)

                # avg_request.success += request_total.success
                # avg_request.fail += request_total.fail
                avg_request.success += cutoff_success
                avg_request.fail += cutoff_fail

                new_rate = get_success_ratio(instance_path)
                # print("New success rate {}%".format(new_rate))
                # print(request_total)
                # print(region_result)
                # print(over_time_result)
                over_time_dict = {}
                for run, region_dict in over_time_result.items():
                    over_time_dict[run] = Request.Request()
                    for region, request in region_dict.items():
                        over_time_dict[run].success += request.success
                        over_time_dict[run].fail += request.fail
                # print(over_time_dict)

            avg_request.success = int(avg_request.success / len(instances))
            avg_request.fail = int(avg_request.fail / len(instances))
            print(avg_request)


def compute_result_single_instance():
    # algorithms = ["RDIFF", "RC-DIFF"]
    algorithms = ["RC-DIFF"]

    threshold_path = "/Users/khoihd/Downloads/hardcode_comparison_new_config"
    agent = 15
    # (topology, instance) = ("scale-free-tree", 0)
    (topology, instance) = ("random-network", 4)
    # (topology, instance, datacenter) = ("random-network", 1, 'B')

    # # remove current output folders and get it from yachay
    # for alg in algorithms:
    #     directory = path + "/" + alg + "/scenario/random-network/d10"
    #     remove_cmd = 'rm -rf ' + directory + "/1"
    #     print(remove_cmd)
    #     scp_cmd = 'scp -r ' + yachay_path + "/" + alg + "/scenario/random-network/d10/1 " + directory
    #     print(scp_cmd)
    #     os.system(remove_cmd)
    #     os.system(scp_cmd)

    # read_and_plot_requests()
    # read_and_print_requests()
    # requests_dict = dict()

    # for threshold_val in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    # topology = 'random-network'
    # topology = 'scale-free-tree'
    # for threshold_val in [0.5]:
    for threshold_val in ["0.7"]:
        threshold = str(threshold_val)
        # plot demand values over time
        request_all_instances = Request.Request()
        for alg in algorithms:
            # for instance in range(0):
            for instance in [4]:
                instance_path = threshold_path + "/" + alg + "/scenario/" + topology + "/d" + str(agent) + "/" + str(instance)
                print(instance_path)
                unzip_output(instance_path, alg)
                demand = read_and_plot_demand(alg, threshold, instance_path)
                demand['Total'] = demand.sum(1)
                print(demand)
                fig, ax = plt.subplots()
                ax.plot(demand)
                ax.set_xticks(range(0, 15))
                ax.set_xticklabels(range(1, 16), fontsize=10)
                ax.set_xlabel("DCOP Run")
                ax.set_ylabel("Demand value")
                ax.legend(demand.columns)
                ax.set_title("Demand over time for " + alg + " with threshold=" + threshold)
                fig.savefig("demand_plot_" + alg + "_" + threshold + ".pdf")
                demand.to_csv("demand_" + alg + "_" + threshold + ".csv")


            # get the number of successful / failed requests
            # for alg in algorithms:
            #     instance_path = threshold_path + "threshold=" + threshold + "/" + alg + "/scenario/" + topology + "/d" + str(agent) + "/" + str(instance)
                print(instance_path)
                _, request_regions, request_total, request_overtime = getRequestsOverTime(alg, instance_path)

                df_request = pd.DataFrame(request_overtime)
                df_request.index = "Region " + df_request.index
                df_request.T.to_csv(alg + ".csv")
                for run, info in request_overtime.items():
                    print("============================")
                    print("run =", run)
                    print("info =", info)
                    success = 0
                    fail = 0
                    for region, request in info.items():
                        success += request.success
                        fail += request.fail
                    print("success =", success)
                    print("fail =", fail)
                    print("total =", success + fail)

                print(request_regions)
                print(request_total)
                print(request_total.success / (request_total.success + request_total.fail) * 100, "%")

                request_all_instances.success += request_total.success
                request_all_instances.fail += request_total.fail


if __name__ == "__main__":
    compute_alg_capacity()
    # compute_result_single_instance
    # compute_result_table()
    # plot_topology = "/Users/khoihd/Downloads/test_ordering_tuple/scenario-0-link-delays/topology.ns"
    # graph = generate_plot(plot_topology)