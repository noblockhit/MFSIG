import plotly.graph_objects as go
import pickle
import numpy as np


class Cluster:
    point_cluster_dict = {}
    clusters = []
    name_counter = 0
    def __init__(self, a, b):
        self.points = [a, b]
        self.name = Cluster.name_counter
        Cluster.name_counter += 1
        Cluster.clusters.append(self)
        Cluster.point_cluster_dict[a] = self
        Cluster.point_cluster_dict[b] = self

    def add_point(self, p):
        self.points.append(p)
        Cluster.point_cluster_dict[p] = self

    def merge(self, other):
        for p in other.points:
            if p not in self.points:
                self.points.append(p)
            Cluster.point_cluster_dict[p] = self

        Cluster.clusters.remove(other)
    
    @classmethod
    def add_new_line(cls, l):
        a, b = l
        cluster_from_a = Cluster.point_cluster_dict.get(a)
        cluster_from_b = Cluster.point_cluster_dict.get(b)

        if cluster_from_a is None and cluster_from_b is None:
            Cluster(a, b)
        
        elif cluster_from_a is not None and cluster_from_b is None:
            cluster_from_a.add_point(b)

        elif cluster_from_b is not None and cluster_from_a is None:
            cluster_from_b.add_point(a)
        
        elif cluster_from_a is cluster_from_b:
            pass
        else:
            cluster_from_a.merge(cluster_from_b)
    
    
    # def __repr__(self) -> str:
    #     return f"Cluster<{self.points}>"
    
    def __repr__(self) -> str:
        return f"Cluster <{self.name}>"
        

def count(circles=None, lines=None):
    if circles is None:
        circles = [[], [], []]
    
    if lines is None:
        lines = [[], [], []]

    print(f"There are {len(circles[0])} circles and {len(lines[0])} lines")
    z_lines = [((lines[0][i], lines[1][i], lines[2][i]), (lines[0][i+1], lines[1][i+1], lines[2][i+1])) for i in range(0, len(lines[0]), 3)]
    for idx, line in enumerate(z_lines):
        if line == ((None, None, None), (None, None, None)):
            continue
        Cluster.add_new_line(line)
        
    markers = []
    count = 0
    for c in Cluster.clusters:
        count += 1
        unzipped_points = list(zip(*c.points))
        marker_data = go.Scatter3d(
            x=unzipped_points[0], 
            y=unzipped_points[1], 
            z=unzipped_points[2], 
            marker=go.scatter3d.Marker(size=3), 
            opacity=0.8, 
            mode='markers'
        )
        markers.append(marker_data)
        
    fig=go.Figure(data=markers)
    fig.write_html('grouped_points.html', auto_open=True)

    print(count)

if __name__ == "__main__":
    count(**pickle.load(open("tmp.3dgraph", "rb")))
    # count(None, lines=[
    #     [0, 10, None, 50, 15, None],
    #     [0, 0, None, 0, 0, None],
    #     [0, 0, None, 0, 0, None]])