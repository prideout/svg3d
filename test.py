#!/usr/bin/env python3

import svg3d
import pyrr
import numpy as np
import svgwrite.utils

from math import *

sign = np.sign
create_ortho = pyrr.matrix44.create_orthogonal_projection
create_perspective = pyrr.matrix44.create_perspective_projection
create_lookat = pyrr.matrix44.create_look_at


def main():
    create_simple_shapes()
    create_complex_shapes()
    generate_octahedra()
    generate_overlapping_triangles()


def generate_overlapping_triangles():
    X = 1
    Y = 2

    view = create_lookat(eye=[1.5, 1.5, 5], target=[1.5, 1.5, 0], up=[0, 1, 0])
    projection = create_ortho(-X, X, -Y, Y, 0, 10)
    pick = pyrr.matrix44.create_from_translation([1, 0, 0])
    left_camera = svg3d.Camera(view, np.dot(projection, pick))

    projection = create_ortho(-X, X, -Y, Y, 0, 10)
    pick = pyrr.matrix44.create_from_translation([-1, 0, 0])
    right_camera = svg3d.Camera(view, np.dot(pick, projection))

    z0 = 0.00
    z1 = 0.01
    z2 = 0.02
    z3 = -0.03

    faces = np.float32(
        [
            [(0, 0, z0), (1, 0, z0), (0.5, 3, z0)],
            [(0, 2, z1), (0, 3, z1), (3, 2.5, z1)],
            [(2, 3, z2), (3, 3, z2), (2.5, 0, z2)],
            [(3, 0, z3), (3, 1, z3), (0, 0.5, z3)],
        ]
    )

    poly_style = dict(
        fill="#e0e0e0",
        fill_opacity="0.75",
        stroke="black",
        stroke_linejoin="round",
        stroke_width="0.01",
    )
    left_scene = svg3d.Scene([svg3d.Mesh(faces, style=poly_style)])
    left_view = svg3d.View(
        left_camera, left_scene, svg3d.Viewport.from_string("-.5 -.5 .5 1")
    )

    z3 = 0.03

    faces = np.float32(
        [
            [(0, 0, z0), (1, 0, z0), (0.5, 3, z0)],
            [(0, 2, z1), (0, 3, z1), (3, 2.5, z1)],
            [(2, 3, z2), (3, 3, z2), (2.5, 0, z2)],
            [(3, 0, z3), (3, 1, z3), (0, 0.5, z3)],
        ]
    )

    poly_style = dict(
        # fill="red",
        fill="#e0e0e0",
        fill_opacity="0.75",
        stroke="black",
        stroke_linejoin="round",
        stroke_width="0.01",
    )
    right_scene = svg3d.Scene([svg3d.Mesh(faces, style=poly_style)])
    right_view = svg3d.View(
        right_camera, right_scene, svg3d.Viewport.from_string("0 -.5 .5 1")
    )

    svg3d.Engine([left_view, right_view]).render("overlapping_triangles.svg")


def generate_octahedra():
    view = create_lookat(eye=[-15, -60, 120], target=[0, 0, 0], up=[0, 1, 0])
    projection = create_perspective(fovy=15, aspect=1, near=10, far=200)
    camera = svg3d.Camera(view, projection)

    verts, faces = icosahedron()
    verts, faces = np.float32(verts), np.uint32(faces)
    faces = 15 * verts[faces]

    centroids = []
    for face in faces:
        centroid = np.float32([0, 0, 0])
        for vert in face:
            centroid += vert
        centroid /= len(face)
        centroids.append(centroid)
    centroids = np.float32([centroids])

    point_style = dict(fill="black", fill_opacity="0.75", stroke="none")
    poly_style = dict(
        fill="#f0f0f0",
        fill_opacity="0.75",
        stroke="black",
        stroke_linejoin="round",
        stroke_width="0.01",
    )

    left_viewport = svg3d.Viewport.from_string("-1.0 -0.5 1.0 1.0")
    right_viewport = svg3d.Viewport.from_string("0.0 -0.5 1.0 1.0")

    back_shader = lambda face_index, winding: None if winding >= 0 else poly_style
    front_shader = lambda face_index, winding: None if winding < 0 else poly_style

    two_pass_scene = svg3d.Scene([])
    two_pass_scene.add_mesh(svg3d.Mesh(faces, back_shader))
    two_pass_scene.add_mesh(svg3d.Mesh(faces, front_shader))
    two_pass_scene.add_mesh(
        svg3d.Mesh(centroids, style=point_style, circle_radius=0.005)
    )

    one_pass_scene = svg3d.Scene([])
    one_pass_scene.add_mesh(svg3d.Mesh(faces, style=poly_style))
    one_pass_scene.add_mesh(
        svg3d.Mesh(centroids, style=point_style, circle_radius=0.005)
    )

    view0 = svg3d.View(camera, one_pass_scene, left_viewport)
    view1 = svg3d.View(camera, two_pass_scene, right_viewport)

    svg3d.Engine([view0, view1]).render(
        "octahedra.svg", (512, 256), "-1.0 -0.5 2.0 1.0"
    )


def rgb(r, g, b):
    r = max(0.0, min(r, 1.0))
    g = max(0.0, min(g, 1.0))
    b = max(0.0, min(b, 1.0))
    return svgwrite.utils.rgb(r * 255, g * 255, b * 255)


def cube():
    return np.float32(
        [
            [(-10, +10, -10), (+10, +10, -10), (+10, -10, -10), (-10, -10, -10)],
            [(-10, +10, +10), (+10, +10, +10), (+10, -10, +10), (-10, -10, +10)],
            [(-10, -10, +10), (+10, -10, +10), (+10, -10, -10), (-10, -10, -10)],
            [(-10, +10, +10), (+10, +10, +10), (+10, +10, -10), (-10, +10, -10)],
            [(-10, -10, +10), (-10, +10, +10), (-10, +10, -10), (-10, -10, -10)],
            [(+10, -10, +10), (+10, +10, +10), (+10, +10, -10), (+10, -10, -10)],
        ]
    )


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


def parametric_surface(slices, stacks, func):
    verts = []
    for i in range(slices + 1):
        theta = i * pi / slices
        for j in range(stacks):
            phi = j * 2.0 * pi / stacks
            p = func(theta, phi)
            verts.append(p)
    verts = np.float32(verts)

    faces = []
    v = 0
    for i in range(slices):
        for j in range(stacks):
            next = (j + 1) % stacks
            faces.append((v + j, v + j + stacks, v + next + stacks, v + next))
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


def mobius_tube(u, v):
    R = 1.5
    n = 3
    u = u * 2
    x = (
        1.0 * R
        + 0.125 * sin(u / 2) * pow(abs(sin(v)), 2 / n) * sign(sin(v))
        + 0.5 * cos(u / 2) * pow(abs(cos(v)), 2 / n) * sign(cos(v))
    ) * cos(u)
    y = (
        1.0 * R
        + 0.125 * sin(u / 2) * pow(abs(sin(v)), 2 / n) * sign(sin(v))
        + 0.5 * cos(u / 2) * pow(abs(cos(v)), 2 / n) * sign(cos(v))
    ) * sin(u)
    z = -0.5 * sin(u / 2) * pow(abs(cos(v)), 2 / n) * sign(cos(v)) + 0.125 * cos(
        u / 2
    ) * pow(abs(sin(v)), 2 / n) * sign(sin(v))
    return x, y, z


def create_simple_shapes():
    view = create_lookat(eye=[50, 40, 120], target=[0, 0, 0], up=[0, 1, 0])
    projection = create_perspective(fovy=15, aspect=1, near=10, far=200)
    camera = svg3d.Camera(view, projection)
    thin_style = dict(
        fill="white",
        stroke="black",
        stroke_linejoin="round",
        fill_opacity="0.75",
        stroke_width="0.002",
    )
    thick_style = dict(
        fill="white",
        stroke="black",
        stroke_linejoin="round",
        fill_opacity="0.75",
        stroke_width="0.005",
    )

    left_viewport = svg3d.Viewport.from_string("-1.0 -0.5 1.0 1.0")
    right_viewport = svg3d.Viewport.from_string("0.0 -0.5 1.0 1.0")

    # Octahedron

    style = dict(
        fill="white",
        fill_opacity="0.75",
        stroke="black",
        stroke_linejoin="round",
        stroke_width="0.005",
    )
    mesh = svg3d.Mesh(15.0 * octahedron(), style=style)
    view = svg3d.View(camera, svg3d.Scene([mesh]))
    svg3d.Engine([view]).render("octahedron.svg")

    # Sphere and Klein

    def shader(face_index, winding):
        return dict(
            fill="white",
            fill_opacity="0.75",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.002",
        )

    slices, stacks = 32, 32
    faces = 15.0 * parametric_surface(slices, stacks, sphere)
    sphere_view = svg3d.View(
        camera, svg3d.Scene([svg3d.Mesh(faces, shader)]), left_viewport
    )

    klein_view = create_lookat(eye=[50, 120, 50], target=[0, 0, 0], up=[0, 0, 1])
    klein_projection = create_perspective(fovy=28, aspect=1, near=10, far=200)
    klein_camera = svg3d.Camera(klein_view, klein_projection)

    faces = 3.0 * parametric_surface(slices, stacks, klein)
    klein_view = svg3d.View(
        klein_camera, svg3d.Scene([svg3d.Mesh(faces, shader)]), right_viewport
    )

    svg3d.Engine([sphere_view, klein_view]).render(
        "sphere_and_klein.svg", (512, 256), "-1.0 -0.5 2.0 1.0"
    )


def create_complex_shapes():
    projection = create_perspective(fovy=25, aspect=1, near=10, far=200)
    view = create_lookat(eye=[25, 20, 60], target=[0, 0, 0], up=[0, 1, 0])
    camera = svg3d.Camera(view, projection)

    # Parametric Sphere

    slices, stacks, radius = 64, 64, 12
    faces = radius * parametric_surface(slices, stacks, sphere)

    antialiasing = "auto"  # use 'crispEdges' to fix cracks

    def shader(face_index, winding):
        slice = int(face_index / 64)
        stack = int(face_index % 64)
        if slice % 3 == 0 or stack % 3 == 0:
            return dict(
                fill="black",
                fill_opacity="1.0",
                stroke="none",
                shape_rendering=antialiasing,
            )
        return dict(
            fill="white",
            fill_opacity="0.75",
            stroke="none",
            shape_rendering=antialiasing,
        )

    scene = svg3d.Scene([svg3d.Mesh(faces, shader)])
    svg3d.Engine([svg3d.View(camera, scene)]).render("parametric_sphere.svg")

    # Sphere Shell

    verts, faces = icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    verts, faces = np.float32(verts), np.int32(faces)
    faces = verts[faces]

    def backface_shader(face_index, winding):
        if winding >= 0:
            return None
        return dict(
            fill="#7f7fff",
            fill_opacity="1.0",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.001",
            stroke_dasharray="0.01",
        )

    def frontface_shader(face_index, winding):
        if winding < 0 or faces[face_index][0][2] > 0.9:
            return None
        return dict(
            fill="#7fff7f",
            fill_opacity="0.6",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.003",
        )

    scene = svg3d.Scene([])
    scene.add_mesh(svg3d.Mesh(12.0 * faces, backface_shader))
    scene.add_mesh(svg3d.Mesh(12.0 * faces, frontface_shader))
    svg3d.Engine([svg3d.View(camera, scene)]).render("sphere_shell.svg")

    # Sphere Lighting

    ones = np.ones(faces.shape[:2] + (1,))
    eyespace_faces = np.dstack([faces, ones])
    eyespace_faces = np.dot(eyespace_faces, view)[:, :, :3]
    shininess = 100
    L = pyrr.vector.normalize(np.float32([20, 20, 50]))
    E = np.float32([0, 0, 1])
    H = pyrr.vector.normalize(L + E)

    def frontface_shader(face_index, winding):
        if winding < 0:
            return None
        face = eyespace_faces[face_index]
        p0, p1, p2 = face[0], face[1], face[2]
        N = pyrr.vector.normalize(pyrr.vector3.cross(p1 - p0, p2 - p0))
        df = max(0, np.dot(N, L))
        sf = pow(max(0, np.dot(N, H)), shininess)
        color = df * np.float32([1, 1, 0]) + sf * np.float32([1, 1, 1])
        color = np.power(color, 1.0 / 2.2)
        return dict(
            fill=rgb(*color), fill_opacity="1.0", stroke="black", stroke_width="0.001"
        )

    scene = svg3d.Scene([])
    scene.add_mesh(svg3d.Mesh(12.0 * faces, frontface_shader))
    svg3d.Engine([svg3d.View(camera, scene)]).render("sphere_lighting.svg")

    # Mobius Tube

    slices, stacks, radius = 48, 32, 7
    faces = radius * parametric_surface(slices, stacks, mobius_tube)

    ones = np.ones(faces.shape[:2] + (1,))
    eyespace_faces = np.dstack([faces, ones])
    eyespace_faces = np.dot(eyespace_faces, view)[:, :, :3]
    shininess = 75
    L = pyrr.vector.normalize(np.float32([10, -10, 50]))
    E = np.float32([0, 0, 1])
    H = pyrr.vector.normalize(L + E)

    def frontface_shader(face_index, winding):
        if winding < 0:
            return None
        face = eyespace_faces[face_index]
        p0, p1, p2 = face[0], face[1], face[2]
        N = pyrr.vector.normalize(pyrr.vector3.cross(p1 - p0, p2 - p0))
        df = max(0, np.dot(N, L))
        sf = pow(max(0, np.dot(N, H)), shininess)
        color = df * np.float32([0, 0.8, 1]) + sf * np.float32([1, 1, 1])
        color = np.power(color, 1.0 / 2.2)
        return dict(
            fill=rgb(*color),
            fill_opacity="1.0",
            stroke=rgb(*(color * 1.5)),
            stroke_width="0.001",
        )

    scene = svg3d.Scene([])
    scene.add_mesh(svg3d.Mesh(faces, frontface_shader))
    svg3d.Engine([svg3d.View(camera, scene)]).render("mobius_tube.svg")

    # Filmstrip

    def shader(face_index, winding):
        return dict(
            fill="white",
            fill_opacity="0.75",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.005",
        )

    thin = dict(
        fill="white",
        fill_opacity="0.75",
        stroke="black",
        stroke_linejoin="round",
        stroke_width="0.001",
    )

    view = create_lookat(eye=[50, 40, 120], target=[0, 0, 0], up=[0, 1, 0])
    projection = create_perspective(fovy=15, aspect=1, near=10, far=200)
    camera = svg3d.Camera(view, projection)

    viewport0 = svg3d.Viewport.from_string("-2.5 -0.5 1.0 1.0")
    viewport1 = svg3d.Viewport.from_string("-1.5 -0.5 1.0 1.0")
    viewport2 = svg3d.Viewport.from_string("-0.5 -0.5 1.0 1.0")
    viewport3 = svg3d.Viewport.from_string(" 0.5 -0.5 1.0 1.0")
    viewport4 = svg3d.Viewport.from_string(" 1.5 -0.5 1.0 1.0")

    slices, stacks = 24, 32
    sphere_faces = 15.0 * parametric_surface(slices, stacks, sphere)

    slices, stacks = 32, 24
    klein_faces = 3.0 * parametric_surface(slices, stacks, klein)

    slices, stacks, radius = 48, 32, 7
    mobius_faces = radius * parametric_surface(slices, stacks, mobius_tube)

    # cube
    view0 = svg3d.View(camera, svg3d.Scene([svg3d.Mesh(cube(), shader)]), viewport0)

    # octahedron
    view1 = svg3d.View(
        camera, svg3d.Scene([svg3d.Mesh(12.0 * octahedron(), shader)]), viewport1
    )

    # sphere
    view2 = svg3d.View(
        camera, svg3d.Scene([svg3d.Mesh(sphere_faces, style=thin)]), viewport2
    )

    # klein
    klein_view = create_lookat(eye=[50, 120, 50], target=[0, 0, 0], up=[0, 0, 1])
    klein_projection = create_perspective(fovy=28, aspect=1, near=10, far=200)
    klein_camera = svg3d.Camera(klein_view, klein_projection)
    view3 = svg3d.View(
        klein_camera, svg3d.Scene([svg3d.Mesh(klein_faces, style=thin)]), viewport3
    )

    # mobius
    view4 = svg3d.View(
        camera, svg3d.Scene([svg3d.Mesh(mobius_faces, frontface_shader)]), viewport4
    )

    drawing = svgwrite.Drawing(
        "filmstrip.svg", (256 * 5, 256), viewBox="-2.5 -0.5 5.0 1.0"
    )
    svg3d.Engine([view0, view1, view2, view3, view4]).render_to_drawing(drawing)
    drawing.save()


main()
