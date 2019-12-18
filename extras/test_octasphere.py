#!/usr/bin/env python3

import numpy as np
import svgwrite.utils
from octasphere import octasphere
import pyrr
from parent_folder import svg3d
from math import *

create_ortho = pyrr.matrix44.create_orthogonal_projection
create_perspective = pyrr.matrix44.create_perspective_projection
create_lookat = pyrr.matrix44.create_look_at

np.set_printoptions(formatter={'float': lambda x: "{0:+0.3f}".format(x)})

quaternion = pyrr.quaternion

SHININESS = 100
DIFFUSE = np.float32([1.0, 0.8, 0.2])
SPECULAR = np.float32([0.5, 0.5, 0.5])
SIZE = (512, 256)

def rgb(r, g, b):
    r = max(0.0, min(r, 1.0))
    g = max(0.0, min(g, 1.0))
    b = max(0.0, min(b, 1.0))
    return svgwrite.utils.rgb(r * 255, g * 255, b * 255)

def rotate_faces(faces):
    q = quaternion.create_from_eulers([pi * -0.4, pi * 0.9, 0])
    new_faces = []
    for f in faces:
        verts = [quaternion.apply_to_vector(q, v) for v in f]
        new_faces.append(verts)
    return np.float32(new_faces)

def translate_faces(faces, offset):
    return faces + np.float32(offset)

def merge_faces(faces0, faces1):
    return np.vstack([faces0, faces1])

projection = create_perspective(fovy=25, aspect=2, near=10, far=200)
view_matrix = create_lookat(eye=[25, 20, 60], target=[0, 0, 0], up=[0, 1, 0])
camera = svg3d.Camera(view_matrix, projection)

def make_octaspheres(ndivisions: int, radius: float, width=0, height=0, depth=0):
    verts, indices = octasphere(ndivisions, radius, width, height, depth)
    faces = verts[indices]

    left = translate_faces(faces, [ -12, 0, 0])
    right = translate_faces(rotate_faces(faces), [ 12, 0, 0])
    faces = merge_faces(left, right)

    ones = np.ones(faces.shape[:2] + (1,))
    eyespace_faces = np.dstack([faces, ones])
    eyespace_faces = np.dot(eyespace_faces, view_matrix)[:, :, :3]
    L = pyrr.vector.normalize(np.float32([20, 20, 50]))
    E = np.float32([0, 0, 1])
    H = pyrr.vector.normalize(L + E)

    def frontface_shader(face_index, winding):
        if winding < 0:
            return None
        face = eyespace_faces[face_index]
        p0, p1, p2 = face[0], face[1], face[2]
        N = pyrr.vector3.cross(p1 - p0, p2 - p0)
        l2 = pyrr.vector3.squared_length(N)
        if l2 > 0:
            N = N / np.sqrt(l2)
        df = max(0, np.dot(N, L))
        sf = pow(max(0, np.dot(N, H)), SHININESS)
        color = df * DIFFUSE + sf * SPECULAR
        color = np.power(color, 1.0 / 2.2)
        return dict(fill=rgb(*color), stroke="black", stroke_width="0.001")

    print(f"Generated octasphere: {ndivisions}, {radius}, {width}, {height}, {depth}")
    return [svg3d.Mesh(faces, frontface_shader)]

vp = svg3d.Viewport(-1, -.5, 2, 1)
engine = svg3d.Engine([])

if False:
    mesh = make_octaspheres(ndivisions=2, radius=8)
    engine.views = [svg3d.View(camera, svg3d.Scene(mesh), vp)]
    engine.render("octasphere3.svg", size=SIZE)

    mesh = make_octaspheres(ndivisions=3, radius=7, width=16, height=16, depth=16)
    engine.views = [svg3d.View(camera, svg3d.Scene(mesh), vp)]
    engine.render("octasphere1.svg", size=SIZE)

    mesh = make_octaspheres(ndivisions=0, radius=7, width=16, height=16, depth=16)
    engine.views = [svg3d.View(camera, svg3d.Scene(mesh), vp)]
    engine.render("octasphere2.svg", size=SIZE)

    mesh = make_octaspheres(ndivisions=3, radius=3, width=12, height=12, depth=12)
    engine.views = [svg3d.View(camera, svg3d.Scene(mesh), vp)]
    engine.render("octasphere4.svg", size=SIZE)

    mesh = make_octaspheres(ndivisions=3, radius=1, width=12, height=12, depth=12)
    engine.views = [svg3d.View(camera, svg3d.Scene(mesh), vp)]
    engine.render("octasphere5.svg", size=SIZE)

    mesh = make_octaspheres(ndivisions=3, radius=3, width=16, height=16, depth=0)
    engine.views = [svg3d.View(camera, svg3d.Scene(mesh), vp)]
    engine.render("octasphere6.svg", size=SIZE)

    mesh = make_octaspheres(ndivisions=3, radius=3, width=16, height=0, depth=16)
    engine.views = [svg3d.View(camera, svg3d.Scene(mesh), vp)]
    engine.render("octasphere7.svg", size=SIZE)

    mesh = make_octaspheres(ndivisions=3, radius=3, width=0, height=16, depth=16)
    engine.views = [svg3d.View(camera, svg3d.Scene(mesh), vp)]
    engine.render("octasphere8.svg", size=SIZE)

    mesh = make_octaspheres(ndivisions=3, radius=0, width=16, height=16, depth=16)
    engine.views = [svg3d.View(camera, svg3d.Scene(mesh), vp)]
    engine.render("octasphere9.svg", size=SIZE)

def tile():
    verts, indices = octasphere(ndivisions=3, radius=3, width=18, height=18, depth=0)
    view_matrix = create_lookat(eye=[25, 20, 60], target=[0, 0, 0], up=[0, 1, 0])
    faces = verts[indices]
    ones = np.ones(faces.shape[:2] + (1,))
    eyespace_faces = np.dstack([faces, ones])
    eyespace_faces = np.dot(eyespace_faces, view_matrix)[:, :, :3]
    L = pyrr.vector.normalize(np.float32([20, 20, 50]))
    E = np.float32([0, 0, 1])
    H = pyrr.vector.normalize(L + E)
    def frontface_shader(face_index, winding):
        if winding < 0:
            return None
        face = eyespace_faces[face_index]
        p0, p1, p2 = face[0], face[1], face[2]
        N = pyrr.vector3.cross(p1 - p0, p2 - p0)
        l2 = pyrr.vector3.squared_length(N)
        if l2 > 0:
            N = N / np.sqrt(l2)
        df = max(0, np.dot(N, L))
        sf = pow(max(0, np.dot(N, H)), SHININESS)
        color = df * DIFFUSE + sf * SPECULAR
        color = np.power(color, 1.0 / 2.2)
        return dict(fill=rgb(*color), stroke="black", stroke_width="0.001")
    return svg3d.Mesh(faces, frontface_shader)

def rounded_cube():
    verts, indices = octasphere(ndivisions=3, radius=1, width=15, height=15, depth=13)
    view_matrix = create_lookat(eye=[25, 20, 60], target=[0, 0, 0], up=[0, 1, 0])
    faces = verts[indices]
    ones = np.ones(faces.shape[:2] + (1,))
    eyespace_faces = np.dstack([faces, ones])
    eyespace_faces = np.dot(eyespace_faces, view_matrix)[:, :, :3]
    L = pyrr.vector.normalize(np.float32([20, 20, 50]))
    E = np.float32([0, 0, 1])
    H = pyrr.vector.normalize(L + E)
    def frontface_shader(face_index, winding):
        if winding < 0:
            return None
        face = eyespace_faces[face_index]
        p0, p1, p2 = face[0], face[1], face[2]
        N = pyrr.vector3.cross(p1 - p0, p2 - p0)
        l2 = pyrr.vector3.squared_length(N)
        if l2 > 0:
            N = N / np.sqrt(l2)
        df = max(0, np.dot(N, L))
        sf = pow(max(0, np.dot(N, H)), SHININESS)
        color = df * DIFFUSE + sf * SPECULAR
        color = np.power(color, 1.0 / 2.2)
        return dict(fill=rgb(*color), stroke="black", stroke_width="0.001")
    return svg3d.Mesh(faces, frontface_shader)

def capsule():
    verts, indices = octasphere(ndivisions=3, radius=4, width=18, height=0, depth=0)
    view_matrix = create_lookat(eye=[25, 20, 60], target=[0, 0, 0], up=[0, 1, 0])
    faces = verts[indices]
    ones = np.ones(faces.shape[:2] + (1,))
    eyespace_faces = np.dstack([faces, ones])
    eyespace_faces = np.dot(eyespace_faces, view_matrix)[:, :, :3]
    L = pyrr.vector.normalize(np.float32([20, 20, 50]))
    E = np.float32([0, 0, 1])
    H = pyrr.vector.normalize(L + E)
    def frontface_shader(face_index, winding):
        if winding < 0:
            return None
        face = eyespace_faces[face_index]
        p0, p1, p2 = face[0], face[1], face[2]
        N = pyrr.vector3.cross(p1 - p0, p2 - p0)
        l2 = pyrr.vector3.squared_length(N)
        if l2 > 0:
            N = N / np.sqrt(l2)
        df = max(0, np.dot(N, L))
        sf = pow(max(0, np.dot(N, H)), SHININESS)
        color = df * DIFFUSE + sf * SPECULAR
        color = np.power(color, 1.0 / 2.2)
        return dict(fill=rgb(*color), stroke="black", stroke_width="0.001")
    return svg3d.Mesh(faces, frontface_shader)

view_matrix = create_lookat(eye=[25, 20, 60], target=[0, 0, 0], up=[0, 1, 0])
projection = create_perspective(fovy=25, aspect=1, near=10, far=200)
camera = svg3d.Camera(view_matrix, projection)
dx = .9
x = -.5
y = -.15
w, h = 1.3, 1.3
engine.views = [
    svg3d.View(camera, svg3d.Scene([tile()]), svg3d.Viewport(x-1, y-.5, w, h)),
    svg3d.View(camera, svg3d.Scene([rounded_cube()]), svg3d.Viewport(x-1+dx, y-.5, w, h)),
    svg3d.View(camera, svg3d.Scene([capsule()]), svg3d.Viewport(x-1+dx*2, y-.5, w, h)),
]
engine.render("ThreeCuboids.svg", size=(600, 200))
