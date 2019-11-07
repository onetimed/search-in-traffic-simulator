import networkx as nx
import pickle
import random
import mplleaflet
import csv
from sklearn.cluster import KMeans
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------- GLOBALS ------------------------------------------------------

with open('data/nodes.pickle', 'rb') as nf:
    nodes = pickle.load(nf)
with open('data/drivable_ways.pickle', 'rb') as wf:
    ways = pickle.load(wf)
with open('data/road_duration.pickle', 'rb') as rdf:
    road_duration = pickle.load(rdf)


# -------------------------------------------- OBJECTS ------------------------------------------------------

class CityMap:

    def __init__(self, ways, road_duration, amount_of_clusters):
        self.parkable_roads = set()
        self.graph = self.create_graph(ways, road_duration)
        self.clusters = self.cluster(amount_of_clusters)

    @staticmethod
    def clean_zero_degree(graph): # Out going degree
        unwanted_nodes = list(filter(lambda x: x[1] == 0, list(graph.out_degree())))
        while len(unwanted_nodes) > 0:
            for node in unwanted_nodes:
                graph.remove_node(node[0])
            unwanted_nodes = list(filter(lambda x: x[1] == 0, list(graph.out_degree())))
        return graph

    @staticmethod
    def add_weights(graph, road_duration):
        for edge in graph.edges():
            if edge in road_duration.keys():
                graph[edge[0]][edge[1]]['weight'] = road_duration[edge]['duration']
                graph[edge[0]][edge[1]]['weight_in_traffic'] = road_duration[edge]['duration_in_traffic']
            else:
                graph[edge[0]][edge[1]]['weight'] = road_duration[edge[::-1]]['duration']
                graph[edge[0]][edge[1]]['weight_in_traffic'] = road_duration[edge[::-1]]['duration_in_traffic']
        return graph


    def create_graph(self, ways, road_duration):
        graph = nx.DiGraph()
        for key in ways.keys():
            if 'oneway' not in ways[key]['tags'].keys() or ways[key]['tags']['oneway'] == 'no':
                graph.add_path((ways[key]['way_nodes'])[::-1]) # WITH MIDDLE
            graph.add_path(ways[key]['way_nodes']) # WITH MIDDLE
            if ways[key]['tags']['highway'] in ['residential', 'living_street', 'unclassified']:
                for i in range(len(ways[key]['way_nodes']) - 1):
                    self.parkable_roads.add((ways[key]['way_nodes'][i], ways[key]['way_nodes'][i+1]))
        graph = self.clean_zero_degree(graph)
        graph = max(nx.strongly_connected_component_subgraphs(graph), key=len)
        graph = self.add_weights(graph, road_duration)
        print('Graph created with Nodes: {0:d}, Links: {1:d}'.format(graph.number_of_nodes(), graph.number_of_edges()))
        return graph

    def shortest_path(self, source_node, destination_node, in_traffic=False):
        if in_traffic:
            return nx.shortest_path(self.graph, source=source_node, target=destination_node, weight='weight_in_traffic')
        else:
            return nx.shortest_path(self.graph, source=source_node, target=destination_node, weight='weight')

    def path_duration(self, path, in_traffic=False):
        duration = 0
        for road in path:
            if in_traffic:
                duration += self.graph[road[0]][road[1]]['weight_in_traffic']
            else:
                duration += self.graph[road[0]][road[1]]['weight']
        return duration

    def parkable(self, road):
        return road in self.parkable_roads

    def cluster(self, amount_of_clusters):
        graph_nodes_for_clustering = []
        graph_nodes = np.array(list(self.graph.nodes()))
        for node in graph_nodes:
            graph_nodes_for_clustering.append([nodes[node]['longitude'], nodes[node]['latitude']])
        graph_nodes_for_clustering = np.array(graph_nodes_for_clustering)
        kmeans = KMeans(n_clusters=amount_of_clusters, random_state=0).fit(graph_nodes_for_clustering)
        clusters = []
        for i in range(amount_of_clusters):
            clusters.append(graph_nodes[np.where(kmeans.labels_ == i)[0]])
        return clusters



class Car:

    def __init__(self, car_id, city_graph, in_traffic=False, source=None, target_list=None, min_path_duration=15):
        # min path duration in minutes.
        self.id = car_id
        self.city_graph = city_graph
        self.in_traffic = in_traffic
        self.type = 'Car'
        legal_route = False
        while not legal_route:
            if source is None:
                if target_list is None:
                    source = random.choice(list(self.city_graph.graph.edges()))
                else:
                    source = random.choice(target_list)
                    neighbor = random.choice(list(city_graph.graph.neighbors(source)))
                    source = (source, neighbor)
            while True:
                if target_list is not None:
                    destination = random.choice(target_list)
                    neighbor = random.choice(list(city_graph.graph.neighbors(destination)))
                    destination = (destination, neighbor)
                else:
                    destination = random.choice(list(self.city_graph.graph.edges()))
                if destination[0] != source[1]:
                    break
            self.route = self.city_graph.shortest_path(source[1], destination[0], in_traffic)
            for i in range(len(self.route) - 1):
                self.route[i] = (self.route[i], self.route[i+1])
            self.route.pop()
            self.route.insert(0, source)
            self.route.append(destination)
            self.time_left_on_route = self.city_graph.path_duration([self.route[0]])
            if (self.city_graph.path_duration(self.route) > (min_path_duration * 60)) and self.city_graph.parkable(self.route[-1]):
                legal_route = True
            if target_list is not None:
                legal_route = True

    def arrive_to_destination(self):
        return len(self.route) == 1

    def update(self, time): # Times in seconds
        if self.arrive_to_destination():
            return [self.route[0]]
        routes = []
        while time >= self.time_left_on_route:
            time -= self.time_left_on_route
            routes.append(self.route.pop(0))
            if self.arrive_to_destination():
                routes.append(self.route[0])
                return routes
            self.time_left_on_route = self.city_graph.path_duration([self.route[0]], self.in_traffic)
        self.time_left_on_route -= time
        routes.append(self.route[0])
        return routes


class System(Car):

    def __init__(self, car_id, city_graph, cluster_id, in_traffic=False, source=None, faster_factor=1.0):
        Car.__init__(self, car_id, city_graph, in_traffic, source, city_graph.clusters[cluster_id])
        self.cars_detected_in_this_road = set()
        self.cluster_id = cluster_id
        self.faster_factor = faster_factor
        self.first = False
        self.last = False
        self.before = None
        self.ahead = set()
        self.after = None
        self.type = "System"

    def update(self, time): # Times in seconds
        routes_visited = []
        while time >= self.time_left_on_route:
            time -= self.time_left_on_route
            routes_visited.append(self.route.pop(0))
            if self.arrive_to_destination():
                self.__init__(self.id, self.city_graph, self.cluster_id,  self.in_traffic, self.route[0])
            self.time_left_on_route = (self.city_graph.path_duration([self.route[0]], self.in_traffic) * (1/self.faster_factor))
        self.time_left_on_route -= time
        routes_visited.append(self.route[0]) # Add current route
        return routes_visited

    def detected_already(self, car_id):
        detected = car_id in self.cars_detected_in_this_road
        self.cars_detected_in_this_road.add(car_id)
        return detected


class Simulator:

    def __init__(self, simulation_id, amount_of_cars, amount_of_systems, type_of_simulation, in_traffic=False, with_clusters=False, faster_factor=1.0):
        self.simulation_id = simulation_id
        self.time_seconds_from_beginning = 0
        self.time_minutes_from_beginning = 0
        self.time_hours_from_beginning = 0
        with open('city_graph/{:02d}_systems.pickle'.format(amount_of_systems), 'rb') as cmf:
            self.city_graph = pickle.load(cmf)
        if not with_clusters: # NO clusters
            type_of_simulation += '_no_cluster'
            for i in range(len(self.city_graph.clusters)):
                self.city_graph.clusters[i] = list(self.city_graph.graph.nodes())
        if in_traffic:
            type_of_simulation += '_in_traffic'

        if not os.path.exists('log_path'):
            os.makedirs('log_path')
        if not os.path.exists('log_path/{:02d}_{}'.format(amount_of_systems, type_of_simulation)):
            os.makedirs('log_path/{:02d}_{}'.format(amount_of_systems, type_of_simulation))
        self.log_path = 'log_path/{:02d}_{}'.format(amount_of_systems, type_of_simulation) + '/' + "simulation_{:02d}".format(simulation_id) + '.csv'
        with open(self.log_path, 'w+', newline='') as log_file:
            csv_writer = csv.writer(log_file, delimiter=',')
            csv_writer.writerow(['time from beginning','cars_state_when_detected', 'cars detected'])

        self.sim_data = dict()
        self.cars_detected_by_back_camera = set()
        self.cars_detected_by_front_camera = set()
        self.cars_detected_by_side_cameras = set()
        self.cars_detected_by_parallel_road = set()
        self.cars = []
        print('creating car')
        for i in range(amount_of_cars):
            self.cars.append(Car(i, self.city_graph, in_traffic))
        self.systems = []
        print('creating system')
        for i in range(amount_of_cars, amount_of_cars+amount_of_systems):
            self.systems.append(System(i, self.city_graph, (i-amount_of_cars), in_traffic, faster_factor=faster_factor))

    def update_time(self, time):
        self.time_seconds_from_beginning += time
        if self.time_seconds_from_beginning >= 60:
            self.time_minutes_from_beginning += (self.time_seconds_from_beginning // 60)
            self.time_seconds_from_beginning %= 60
            if self.time_minutes_from_beginning >= 60:
                self.time_hours_from_beginning += (self.time_minutes_from_beginning // 60)
                self.time_minutes_from_beginning %= 60

    def time_string(self):
        return "{:02d}:{:02d}:{:02d}".format(self.time_hours_from_beginning, self.time_minutes_from_beginning, self.time_seconds_from_beginning)

    def update_front_camera(self, system, road):
        if system.first or system.before is not None:
            return
        possible = self.sim_data[road]['opposite direction'].union(self.sim_data[road]['driving'])
        possible.add(None)
        system.before = random.choice(list(possible))
        possible.remove(system.before)
        if system.before is None:
            system.first = True
        else:
            self.cars_detected_by_front_camera.add((system.id, system.before.id))
        return possible

    def update_back_camera(self, system, road):
        if system.after is not None:
            return
        possible = self.sim_data[road]['opposite direction'].union(self.sim_data[road]['driving'])
        possible.add(None)
        if system.before in possible:
            possible.remove(system.before)
        if system.last:
            possible.remove(None)
            possible -= system.ahead
            if len(possible) != 0:
                system.after = random.choice(list(possible))
                system.last = False
                self.cars_detected_by_back_camera.add((system.id, system.after.id))
            return
        system.after = random.choice(list(possible))
        possible.remove(system.after)
        system.ahead = possible
        if system.after is None:
            system.last = True
        else:
            self.cars_detected_by_back_camera.add((system.id, system.after.id))

    def update_side_camera(self, system, road):
        for car in self.sim_data[road]['parking']:
            if not system.detected_already(car.id):
                self.cars_detected_by_side_cameras.add((system.id, car.id))

    def update_parallel_camera(self, system, road):
        for car in self.sim_data[road]['opposite direction'].union(self.sim_data[road]['driving']):
            if not system.detected_already(car.id):
                self.cars_detected_by_parallel_road.add((system.id, car.id))

    def update_simulator(self, time): # Time in seconds
        self.update_time(time)
        self.sim_data = dict()
        for car in self.cars:
            roads_passed = car.update(time)

            for i, road in enumerate(roads_passed):
                if road not in self.sim_data.keys():
                    self.sim_data[road] = dict()
                    self.sim_data[road]['parking'] = set()
                    self.sim_data[road]['driving'] = set()
                    self.sim_data[road]['opposite direction'] = set()
                if i == (len(roads_passed) - 1):
                    if car.arrive_to_destination():  # CAR ALREADY PARKED
                        self.sim_data[road]['parking'].add(car)
                    else:
                        self.sim_data[road]['driving'].add(car)
                else:
                    self.sim_data[road]['opposite direction'].add(car)

        for system in self.systems:
            roads_visited = system.update(time)

            for i, road in enumerate(roads_visited):
                if road in self.sim_data.keys():
                    self.update_side_camera(system, road)
                    if (len(self.sim_data[road]['driving']) != 0) or (len(self.sim_data[road]['opposite direction']) != 0):
                        self.update_front_camera(system, road)
                        self.update_back_camera(system, road)
                if road[::-1] in self.sim_data.keys():
                    self.update_side_camera(system, road[::-1])
                    self.update_parallel_camera(system, road[::-1])
                if i != (len(roads_visited) - 1):
                    system.cars_detected_in_this_road = set()  # Empty detection on this road
                    system.first = False
                    system.last = False
                    system.before = None
                    system.ahead = set()
                    system.after = None

    def log_data(self):
        time = self.time_string()
        with open(self.log_path, 'a', newline='') as log_file:
            csv_writer = csv.writer(log_file, delimiter=',')
            if len(self.cars_detected_by_back_camera) > 0:
                csv_writer.writerow([time,'back camera'] + list(self.cars_detected_by_back_camera))
                self.cars_detected_by_back_camera = set()
            if len(self.cars_detected_by_front_camera) > 0:
                csv_writer.writerow([time,'front camera'] + list(self.cars_detected_by_front_camera))
                self.cars_detected_by_front_camera = set()
            if len(self.cars_detected_by_side_cameras) > 0:
                csv_writer.writerow([time,'side camera'] + list(self.cars_detected_by_side_cameras))
                self.cars_detected_by_side_cameras = set()
            if len(self.cars_detected_by_parallel_road) > 0:
                csv_writer.writerow([time,'front camera parallel road'] + list(self.cars_detected_by_parallel_road))
                self.cars_detected_by_parallel_road = set()

    def display(self):
        plt.figure()
        for car in self.cars:
            node0, node1 = car.route[0]
            if car.arrive_to_destination():
                plt.plot(((nodes[node0]['longitude'] + nodes[node1]['longitude']) / 2), ((nodes[node0]['latitude'] + nodes[node1]['latitude']) / 2), 'gs')
            else:
                plt.plot(((nodes[node0]['longitude'] + nodes[node1]['longitude']) / 2), ((nodes[node0]['latitude'] + nodes[node1]['latitude']) / 2), 'bs')

        for system in self.systems:
            node0, node1 = system.route[0]
            plt.plot(((nodes[node0]['longitude'] + nodes[node1]['longitude']) / 2), ((nodes[node0]['latitude'] + nodes[node1]['latitude']) / 2), 'rs')

        mplleaflet.save_html(fileobj='images/{:02d}{:02d}{:02d}_simulation_{}.html'.format(self.time_hours_from_beginning, self.time_minutes_from_beginning, self.time_seconds_from_beginning, self.simulation_id))

    def run_simulation(self, simulation_time, update_time):
        # simulation time is in hours.
        # update time is in seconds.
        while self.time_hours_from_beginning < simulation_time:
            if self.time_minutes_from_beginning % 10 == 0 and self.time_seconds_from_beginning == 0:
                print(self.time_string())
            self.update_simulator(update_time)
            self.log_data()


if __name__ == '__main__':
    # Example
    rounds = 1000
    for amount_of_systems in [10]:
        for traffic in [False, True]:
            print("{} systems and {} traffic".format(str(amount_of_systems), str(traffic)))
            for j in range(rounds):
                print('starting simulation ', j, ' out of ', rounds)
                sim = Simulator(j, 1, amount_of_systems,'faster_system' , in_traffic=traffic, faster_factor=1.2)
                sim.run_simulation(1, 1)
