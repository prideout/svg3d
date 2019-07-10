#!/usr/bin/env python3

import numpy as np
import svg3d
import pyrr

create_view_matrix = pyrr.matrix44.create_look_at
create_proj_matrix = pyrr.matrix44.create_perspective_projection


def create_meshes(style, shader):
    meshes = []

    # [0,4] wide faces spanning left to right.
    faces = [
        [[ 0.16,  0.09,  0.0], [ 0.13,  0.06, 0.0], [-0.18,  0.09, 0.0]],
        [[ 0.19,  0.13,  0.0], [ 0.16,  0.09, 0.0], [-0.2 ,  0.13, 0.0]],
        [[ 0.2 ,  0.17,  0.0], [ 0.19 , 0.13 , 0.0], [-0.21 , 0.17 , 0.0]],
        [[ 0.21,  0.21,  0.0], [ 0.2  , 0.17 , 0.0], [-0.21 , 0.21 , 0.0]],
        [[ 0.2 ,  0.26,  0.0], [ 0.21 , 0.21 , 0.0], [-0.21 , 0.26 , 0.0]]]

    # [5,14] small upward pointing faces.
    for x in range(10):
        fx = 0.02 + -0.3 * x / 10.0
        faces.append([[0.15+fx, 0.15, 0.0], [0.17+fx, 0.15, 0.0], [0.16+fx, 0.15, -.01]])

    # [15,24] varying-angle upward faces.
    for x in range(10):
        fx = -0.3 * x / 10.0
        fy = 0.02 * (-5 + x)
        y = 0.19
        faces.append([[0.15+fx, y, 0.0], [0.17+fx, y, 0.0], [0.16+fx, y+fy, -.05]])

    # [25] One large face behind everything.
    faces.append([[ 0.16,  0.09,  0.1], [ 0.21,  0.21,  0.1], [-0.19 , 0.35 , 0.1] ])

    faces = np.float32(faces)
    meshes.append(svg3d.Mesh(faces, shader=shader, style=style))

    return meshes


def generate_svg_from_meshes(filename, meshes):
    view = create_view_matrix(eye=[1, -1, -2], target=[0, 0.3, 0], up=[0, 1, 0])
    # view = create_view_matrix(eye=[2, 0.15, -0.05], target=[0, 0.3, 0], up=[0, 1, 0]) # This camera position makes it clear there are no self-intersections
    projection = create_proj_matrix(fovy=15, aspect=1, near=10, far=100)
    # z = 0.25; projection = pyrr.matrix44.create_orthogonal_projection(-z, z, z, -z, z, 100)
    camera = svg3d.Camera(view, projection)
    view = svg3d.View(camera, svg3d.Scene(meshes))
    svg3d.Engine([view]).render(filename, style="background-color: grey; width: 100%; height: 100%;")


style = dict(
    fill_opacity="0.75",
    stroke="black",
    stroke_linejoin="round",
    stroke_width="0.002",
)

front_style = dict(fill="#e0efe0")
back_style = dict(fill="#efe0e0")

shader = lambda face_index, winding: front_style if winding >= 0 else back_style

meshes = create_meshes(style, shader)
generate_svg_from_meshes("sorted.svg", meshes)
