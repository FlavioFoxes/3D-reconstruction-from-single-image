import trimesh
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

"""
It samples points on the surface of the mesh.
It generates the Ground Truth of out dataset.

Arguments:
        mesh_path (string):     path to the mesh
        num_points:             number of points sampled on the surface
                                of the mesh
"""
def mesh_to_point_cloud(mesh_path, num_points=1024):
    # Load the mesh
    mesh = trimesh.load(mesh_path, force='mesh')

    # Sample points on the surface of the mesh
    points, _ = trimesh.sample.sample_surface(mesh, num_points)

    return points
"""
Returns the trace of the figure. 
It must be added manually to the figure to be displayed

Argument:
        points:         points to be plotted

"""
def display_point_cloud(points):
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    trace = go.Scatter3d(
        x = x, y = y, z = z,mode = 'markers', marker = dict(
            size = 7
            # color = z,
            # colorscale = 'Viridis'
        )
    )
    return trace
