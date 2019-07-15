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
    width: float = 1.0
    height: float = 1.0

    @classmethod
    def from_aspect(cls, aspect_ratio: float):
        return cls(-aspect_ratio / 2.0, -0.5, aspect_ratio, 1.0)

    @classmethod
    def from_string(cls, string_to_parse):
        args = [float(f) for f in string_to_parse.split()]
        return cls(*args)


class Camera(NamedTuple):
    view: np.ndarray
    projection: np.ndarray


class Mesh(NamedTuple):
    faces: np.ndarray
    shader: Callable[[int, float], dict] = None
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
    def __init__(self, views, precision=5):
        self.views = views
        self.precision = precision

    def render(self, filename, size=(512, 512), viewBox="-0.5 -0.5 1.0 1.0", **extra):
        drawing = svgwrite.Drawing(filename, size, viewBox=viewBox, **extra)
        self.render_to_drawing(drawing)
        drawing.save()

    def render_to_drawing(self, drawing):
        for view in self.views:
            projection = np.dot(view.camera.view, view.camera.projection)

            clip_path = drawing.defs.add(drawing.clipPath())
            clip_min = view.viewport.minx, view.viewport.miny
            clip_size = view.viewport.width, view.viewport.height
            clip_path.add(drawing.rect(clip_min, clip_size))

            for mesh in view.scene.meshes:
                g = self._create_group(drawing, projection, view.viewport, mesh)
                g["clip-path"] = clip_path.get_funciri()
                drawing.add(g)

    def _create_group(self, drawing, projection, viewport, mesh):
        faces = mesh.faces
        shader = mesh.shader or (lambda face_index, winding: {})
        default_style = mesh.style or {}

        # Extend each point to a vec4, then transform to clip space.
        faces = np.dstack([faces, np.ones(faces.shape[:2])])
        faces = np.dot(faces, projection)

        # Reject trivially clipped polygons.
        w = faces[:, :, 3:4]
        xy = faces[:, :, 0:2]
        accepted = np.logical_and(np.greater(xy, -w), np.less(xy, +w))
        accepted = np.all(accepted, 2)  # vert is accepted if xyz are all inside
        accepted = np.any(accepted, 1)  # face is accepted if any vert is inside
        degenerate = np.less_equal(w, 0)[:, :, 0]  # vert is bad if its w <= 0
        degenerate = np.any(degenerate, 1)  # face is bad if any of its verts are bad
        accepted = np.logical_and(accepted, np.logical_not(degenerate))
        faces = np.compress(accepted, faces, axis=0)

        # Divide X Y Z by W and discard W.
        faces = faces[:, :, :3] / faces[:, :, 3:4]

        # Apply viewport transform to X and Y.
        faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * viewport.width / 2
        faces[:, :, 1:2] = (1.0 - faces[:, :, 1:2]) * viewport.height / 2
        faces[:, :, 0:1] += viewport.minx
        faces[:, :, 1:2] += viewport.miny

        # Sort faces from back to front.
        z_centroids = -np.sum(faces[:, :, 2], axis=1)
        for face_index in range(len(z_centroids)):
            z_centroids[face_index] /= len(faces[face_index])
        face_indices = np.argsort(z_centroids)
        faces = faces[face_indices]

        # Compute the winding direction of each polygon.
        windings = np.zeros(faces.shape[0])
        if faces.shape[1] >= 3:
            p0 = faces[:, 0, :]
            p1 = faces[:, 1, :]
            p2 = faces[:, 2, :]
            normals = np.cross(p2 - p0, p1 - p0)
            np.copyto(windings, normals[:, 2])

        # Determine the style for each polygon and add it to the group.
        group = drawing.g(**default_style)
        for face_index, face in enumerate(faces):
            style = shader(face_indices[face_index], windings[face_index])
            if style is None:
                continue
            face = np.around(face, self.precision)
            if mesh.circle_radius == 0:
                group.add(drawing.polygon(face[:, 0:2], **style))
                continue
            for pt in face:
                group.add(drawing.circle(pt[0:2], mesh.circle_radius, **style))

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
