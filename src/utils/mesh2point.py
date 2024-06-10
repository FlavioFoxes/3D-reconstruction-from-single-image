import trimesh
import matplotlib.pyplot as plt
def mesh_to_point_cloud(mesh_path, num_points=1024):
    # Load the mesh
    mesh = trimesh.load(mesh_path, force='mesh')

    # Sample points on the surface of the mesh
    points, _ = trimesh.sample.sample_surface(mesh, num_points)

    return points

def display_point_cloud(point_cloud):
    # Extract x, y, z coordinates
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud')
    plt.show()

def write_las_file(point_cloud, las_file_path):
    # Create a new LAS file
    outfile = laspy.create(point_format=1)

    # Set up LAS file header
    outfile.header.scale = [0.01, 0.01, 0.01]  # Scale factor for x, y, z coordinates
    outfile.header.offset = [0, 0, 0]  # Offset for x, y, z coordinates
    outfile.header.min = point_cloud.min(axis=0)  # Minimum values of x, y, z coordinates
    outfile.header.max = point_cloud.max(axis=0)  # Maximum values of x, y, z coordinates

    # Add points to the LAS file
    outfile.x = point_cloud[:, 0]
    outfile.y = point_cloud[:, 1]
    outfile.z = point_cloud[:, 2]

    # Close the LAS file
    #outfile.close()
# if __name__ == "__main__":
# Path to the OBJ file
# obj_file_path = "cube.obj"
    
# # Number of points to sample on the surface of the mesh
# num_points = 1000

# # Convert mesh to point cloud
# point_cloud = mesh_to_point_cloud(obj_file_path, num_points)
# print(point_cloud)
# display_point_cloud(point_cloud)
# Path to save the LAS file
# las_file_path = "/home/flavio/Documenti/University/output_point_cloud.las"

# Write the point cloud to a LAS file
# write_las_file(point_cloud, las_file_path)
