# svg3d :: https://prideout.net/blog/svg_wireframes/
# Single-file Python library for generating 3D wireframes in SVG format.
# Copyright (c) 2019 Philip Rideout
# Distributed under the MIT License, see bottom of file.

import numpy as np
import pyrr
import svgwrite

from typing import NamedTuple, Callable, Sequence


class Viewport(NamedTuple):
    minx: float = -0.5
    miny: float = -0.5
    width: float = 1
    height: float = 1

    @classmethod
    def from_string(cls, string_to_parse):
        args = [float(f) for f in string_to_parse.split()]
        return cls(*args)

    def min(self):
        return np.float32([self.minx, self.miny])

    def dims(self):
        return np.float32([self.width, self.height])


class Camera(NamedTuple):
    view: np.ndarray
    projection: np.ndarray


class Mesh(NamedTuple):
    faces: np.ndarray
    shader: Callable[[int, float], dict] = lambda face_index, winding: {}
    style: dict = None
    circle_radius: float = 0


class Scene(NamedTuple):
    meshes: Sequence[Mesh]

    def add_mesh(self, mesh: Mesh):
        self.meshes.append(mesh)


class View(NamedTuple):
    camera: Camera
    scene: Scene
    viewport: Viewport = Viewport()


class Engine:
    def __init__(self, views):
        self.views = views

    def render(self, filename, size=(512, 512), viewBox="-0.5 -0.5 1.0 1.0", **extra):
        drawing = svgwrite.Drawing(filename, size, viewBox=viewBox, **extra)
        self.render_to_drawing(drawing)
        drawing.save()

    def render_to_drawing(self, drawing):
        for view in self.views:
            projection = np.dot(view.camera.view, view.camera.projection)
            for mesh in view.scene.meshes:
                drawing.add(
                    self._create_group(drawing, projection, view.viewport, mesh)
                )

    def _create_group(self, drawing, projection, viewport, mesh):
        faces = mesh.faces

        # Extend each point to a vec4, then multiply by the MVP.
        ones = np.ones(faces.shape[:2] + (1,))
        faces = np.dstack([faces, ones])
        faces = np.dot(faces, projection)

        # Divide X Y Z by W, then discard W.
        faces[:, :, :3] /= faces[:, :, 3:4]
        faces = faces[:, :, :3]

        # Apply viewport transform to X Y.
        faces[:, :, 0:2] = (
            (faces[:, :, 0:2] + 1.0) * viewport.dims() / 2
        ) + viewport.min()

        # Sort faces from back to front.
        z_centroids = -np.sum(faces[:, :, 2], axis=1)
        for face_index in range(len(z_centroids)):
            z_centroids[face_index] /= len(faces[face_index])
        face_indices = np.argsort(z_centroids)
        faces = faces[face_indices]

        # Compute the winding direction of each polygon, determine its
        # style, and add it to the group. If the returned style is None,
        # cull away the polygon.
        if mesh.style == None:
            group = drawing.g()
        else:
            group = drawing.g(**mesh.style)
        face_index = 0
        for face in faces:
            p0, p1, p2 = face[0], face[1], face[2]
            winding = pyrr.vector3.cross(p1 - p0, p2 - p0)[2]
            style = mesh.shader(face_indices[face_index], winding)
            if style != None:
                if mesh.circle_radius == 0:
                    group.add(drawing.polygon(face[:, 0:2], **style))
                else:
                    for pt in face:
                        group.add(drawing.circle(pt[0:2], mesh.circle_radius, **style))
            face_index = face_index + 1

        return group


# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
