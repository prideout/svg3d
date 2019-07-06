#!/usr/bin/env python3

import svg3d
import pyrr
import numpy as np
import svgwrite

from math import *

def rgb(r, g, b):
    return svgwrite.utils.rgb(r * 255, g * 255, b * 255)

def icosahedron():
    """Construct a 20-sided polyhedron"""
    faces = [ (0,1,2), (0,2,3), (0,3,4), (0,4,5), (0,5,1), (11,7,6), (11,8,7), (11,9,8), (11,10,9), (11,6,10), (1,6,2), (2,7,3), (3,8,4), (4,9,5), (5,10,1), (6,7,2), (7,8,3), (8,9,4), (9,10,5), (10,6,1) ]
    verts = [ ( 0.000,  0.000,  1.000 ), ( 0.894,  0.000,  0.447 ), ( 0.276,  0.851,  0.447 ), (-0.724,  0.526,  0.447 ), (-0.724, -0.526,  0.447 ), ( 0.276, -0.851,  0.447 ), ( 0.724,  0.526, -0.447 ), (-0.276,  0.851, -0.447 ), (-0.894,  0.000, -0.447 ), (-0.276, -0.851, -0.447 ), ( 0.724, -0.526, -0.447 ), ( 0.000,  0.000, -1.000 ) ]
    return verts, faces

def subdivide(verts, faces):
    """Subdivide each triangle into four triangles, pushing verts to the unit sphere"""
    triangles = len(faces)
    for faceIndex in range(triangles):
    
        # Create three new verts at the midpoints of each edge:
        face = faces[faceIndex]
        a,b,c = np.float32([verts[vertIndex] for vertIndex in face])
        verts.append(pyrr.vector.normalize(a + b))
        verts.append(pyrr.vector.normalize(b + c))
        verts.append(pyrr.vector.normalize(a + c))

        # Split the current triangle into four smaller triangles:
        i = len(verts) - 3
        j, k = i+1, i+2
        faces.append((i, j, k))
        faces.append((face[0], i, k))
        faces.append((i, face[1], j))
        faces[faceIndex] = (k, j, face[2])

    return verts, faces

def parametric_surface(slices, stacks, func):
    verts = []
    for i in range(slices + 1):
        theta = i * pi / slices
        for j in range(stacks):
            phi = j * 2.0 *  pi / stacks
            p = func(theta, phi)
            verts.append(p)
    verts = np.float32(verts)

    faces = []
    v = 0
    for i in range(slices):
        for j in range(stacks):
            next = (j + 1) % stacks
            faces.append((v + j, v + j + stacks, v + next + stacks, v + next ))
        v = v + stacks
    faces = np.int32(faces)
    return verts[faces]

def sphere(u, v):
    x = sin(u) * cos(v)
    y = cos(u)
    z = -sin(u) * sin(v)
    return x, y, z

def klein(u, v):
    u = u * 2
    if u < pi:
        x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(u) * cos(v)
        z = -8 * sin(u) - 2 * (1 - cos(u) / 2) * sin(u) * cos(v)
    else:
        x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(v + pi)
        z = -8 * sin(u)
    y = -2 * (1 - cos(u) / 2) * sin(v)
    return x, y, z

# Create projection

projection = pyrr.matrix44.create_perspective_projection(fovy=25, aspect=1, near=10, far=100)
view = pyrr.matrix44.create_look_at(eye=[25, -20, 60], target=[0, 0, 0], up=[0, 1, 0])
camera = svg3d.Camera(view, projection)

# Parametric Sphere

slices, stacks, radius = 64, 64, 12
faces = radius * parametric_surface(slices, stacks, sphere)

def shader(face_index, winding):
    slice = int(face_index / 64)
    stack = int(face_index % 64)
    if slice % 3 == 0 or stack % 3 == 0:
        return dict(fill='black', fill_opacity='1.0', stroke='none')
    return dict(fill='white', fill_opacity='0.75', stroke='none')

scene = svg3d.Scene(svg3d.Mesh(faces, shader))
svg3d.Engine(svg3d.View(camera, scene)).render('parametric_sphere.svg')

# Sphere Shell

verts, faces = icosahedron()
verts, faces = subdivide(verts, faces)
verts, faces = subdivide(verts, faces)
verts, faces = np.float32(verts), np.int32(faces)
faces = verts[faces]

def backface_shader(face_index, winding):
    if winding >= 0: return None
    return dict(
        fill='#7f7fff', fill_opacity='1.0',
        stroke='black', stroke_linejoin='round',        
        stroke_width='0.001', stroke_dasharray='0.01')

def frontface_shader(face_index, winding):
    if winding < 0 or faces[face_index][0][2] > 0.9: return None
    return dict(
        fill='#7fff7f', fill_opacity='0.6',
        stroke='black', stroke_linejoin='round',
        stroke_width='0.003')

scene = svg3d.Scene()
scene.add_mesh(svg3d.Mesh(12.0 * faces, backface_shader))
scene.add_mesh(svg3d.Mesh(12.0 * faces, frontface_shader))
svg3d.Engine(svg3d.View(camera, scene)).render('sphere_shell.svg')

# Sphere Lighting

ones = np.ones(faces.shape[:2] + (1,))
eyespace_faces = np.dstack([faces, ones])
eyespace_faces = np.dot(eyespace_faces, view)[:,:,:3]
shininess = 100
L = pyrr.vector.normalize(np.float32([20, -20, 50]))
E = np.float32([0, 0, 1])
H = pyrr.vector.normalize(L + E)

def frontface_shader(face_index, winding):
    if winding < 0: return None
    face = eyespace_faces[face_index]
    p0, p1, p2 = face[0][:], face[1][:], face[2][:]
    N = pyrr.vector.normalize(pyrr.vector3.cross(p1 - p0, p2 - p0))
    df = max(0, np.dot(N, L))
    sf = pow(max(0, np.dot(N, H)), shininess)

    color = df * np.float32([1, 1, 0]) + sf * np.float32([1, 1, 1])
    color[0] = min(1.0, pow(color[0], 1.0 / 2.2))
    color[1] = min(1.0, pow(color[1], 1.0 / 2.2))
    color[2] = min(1.0, pow(color[2], 1.0 / 2.2))

    return dict(
        fill=rgb(*color), fill_opacity='1.0',
        stroke='black', stroke_width='0.001',
        shape_rendering='crispEdges')

scene = svg3d.Scene()
scene.add_mesh(svg3d.Mesh(12.0 * faces, frontface_shader))
svg3d.Engine(svg3d.View(camera, scene)).render('sphere_lighting.svg')
