import numpy as np

def off_to_numpy(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Skip the 'OFF' header
    assert lines[0].strip() == 'OFF'
    
    # Read the number of vertices, faces, and edges
    v_count, f_count, _ = map(int, lines[1].strip().split())
    
    # Read the vertices
    vertices = []
    for i in range(2, 2 + v_count):
        vertices.append(list(map(float, lines[i].strip().split())))
    vertices = np.array(vertices)
    
    # Read the faces
    faces = []
    for i in range(2 + v_count, 2 + v_count + f_count):
        face = list(map(int, lines[i].strip().split()))[1:]  # Remove the first number (count of vertices)
        faces.append(face)
    faces = np.array(faces)
    
    return vertices, faces

# Example usage:
vertices, faces = off_to_numpy('path_to_your_file.off')

print("Vertices:\n", vertices)
print("Faces:\n", faces)
