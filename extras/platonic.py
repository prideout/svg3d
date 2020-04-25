#!/usr/bin/env python3

from parent_folder import svg3d
import pyrr
import numpy as np
import svgwrite.utils

from math import *

sign = np.sign
create_ortho = pyrr.matrix44.create_orthogonal_projection
create_perspective = pyrr.matrix44.create_perspective_projection
create_lookat = pyrr.matrix44.create_look_at
quaternion = pyrr.quaternion


def main():
    create_octahedron_pair("platonic_octahedron.svg") # hexahedron
    # dodecahedron and the icosahedron
    # tetrahedron


def create_octahedron_pair(filename):
    vp = svg3d.Viewport.from_aspect(2)
    projection = create_perspective(fovy=25, aspect=2, near=10, far=200)
    view = create_lookat(eye=[0, 20, 60], target=[0, 0, 0], up=[0, 1, 0])
    camera = svg3d.Camera(view, projection)

    scene = svg3d.Scene([])

    faces = octahedron()
    def backface_shader(face_index, winding):
        if winding >= 0: return None
        return dict(
            fill="#7f7fff",
            fill_opacity="1.0",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.002",
            stroke_dasharray="0.01",
        )
    def frontface_shader(face_index, winding):
        if winding < 0: return None
        return dict(
            fill="#7fff7f",
            fill_opacity="0.4",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.003",
        )
    faces += np.array([-1.25, 0, 0])
    scene.add_mesh(svg3d.Mesh(12.0 * faces, backface_shader))
    scene.add_mesh(svg3d.Mesh(12.0 * faces, frontface_shader))

    faces2 = hexahedron() * 0.8
    def backface_shader2(face_index, winding):
        if winding >= 0: return None
        return dict(
            fill="#7f7fff",
            fill_opacity="1.0",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.002",
            stroke_dasharray="0.01",
        )
    def frontface_shader2(face_index, winding):
        if winding < 0: return None
        return dict(
            fill="#7fff7f",
            fill_opacity="0.4",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.003",
        )

    q = quaternion.create_from_eulers([0, pi * 0.25, 0])
    for f in faces2:
        for v in f:
            v[:] = quaternion.apply_to_vector(q, v)

    faces2 += np.array([12.0, 0, 0])
    scene.add_mesh(svg3d.Mesh(faces2, backface_shader2))
    scene.add_mesh(svg3d.Mesh(faces2, frontface_shader2))

    e = svg3d.Engine([svg3d.View(camera, scene, vp)])
    e.render(filename, (1024, 512))


def subdivide(verts, faces):
    """Subdivide each triangle into four triangles, pushing verts to the unit sphere"""
    triangles = len(faces)
    for faceIndex in range(triangles):

        # Create three new verts at the midpoints of each edge:
        face = faces[faceIndex]
        a, b, c = np.float32([verts[vertIndex] for vertIndex in face])
        verts.append(pyrr.vector.normalize(a + b))
        verts.append(pyrr.vector.normalize(b + c))
        verts.append(pyrr.vector.normalize(a + c))

        # Split the current triangle into four smaller triangles:
        i = len(verts) - 3
        j, k = i + 1, i + 2
        faces.append((i, j, k))
        faces.append((face[0], i, k))
        faces.append((i, face[1], j))
        faces[faceIndex] = (k, j, face[2])

    return verts, faces


def hexahedron():
    verts = np.float32(
        [
            [(-10, +10, -10), (+10, +10, -10), (+10, -10, -10), (-10, -10, -10)],
            [(-10, +10, +10), (+10, +10, +10), (+10, -10, +10), (-10, -10, +10)],
            [(-10, -10, +10), (+10, -10, +10), (+10, -10, -10), (-10, -10, -10)],
            [(-10, +10, +10), (+10, +10, +10), (+10, +10, -10), (-10, +10, -10)],
            [(-10, -10, +10), (-10, +10, +10), (-10, +10, -10), (-10, -10, -10)],
            [(+10, -10, +10), (+10, +10, +10), (+10, +10, -10), (+10, -10, -10)],
        ]
    )
    faces = np.float32([
        verts[0],
        verts[1],
        verts[2],
        verts[3],
        verts[4],
        verts[5]
    ])
    faces[5] = list(reversed(verts[5]))
    faces[1] = list(reversed(verts[1]))
    faces[2] = list(reversed(verts[2]))
    return faces


def octahedron():
    """Construct an eight-sided polyhedron"""
    f = sqrt(2.0) / 2.0
    verts = np.float32(
        [(0, -1, 0), (-f, 0, f), (f, 0, f), (f, 0, -f), (-f, 0, -f), (0, 1, 0)]
    )
    faces = np.int32(
        [
            (0, 2, 1),
            (0, 3, 2),
            (0, 4, 3),
            (0, 1, 4),
            (5, 1, 2),
            (5, 2, 3),
            (5, 3, 4),
            (5, 4, 1),
        ]
    )
    return verts[faces]


def icosahedron():
    """Construct a 20-sided polyhedron"""
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 1),
        (11, 7, 6),
        (11, 8, 7),
        (11, 9, 8),
        (11, 10, 9),
        (11, 6, 10),
        (1, 6, 2),
        (2, 7, 3),
        (3, 8, 4),
        (4, 9, 5),
        (5, 10, 1),
        (6, 7, 2),
        (7, 8, 3),
        (8, 9, 4),
        (9, 10, 5),
        (10, 6, 1),
    ]
    verts = [
        (0.000, 0.000, 1.000),
        (0.894, 0.000, 0.447),
        (0.276, 0.851, 0.447),
        (-0.724, 0.526, 0.447),
        (-0.724, -0.526, 0.447),
        (0.276, -0.851, 0.447),
        (0.724, 0.526, -0.447),
        (-0.276, 0.851, -0.447),
        (-0.894, 0.000, -0.447),
        (-0.276, -0.851, -0.447),
        (0.724, -0.526, -0.447),
        (0.000, 0.000, -1.000),
    ]
    return verts, faces

main()
