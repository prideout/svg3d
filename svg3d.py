# svg3d :: https://prideout.net/blog/svg_wireframes/
# Single-file Python library for generating 3D wireframes in SVG format.
# Copyright (c) 2019 Philip Rideout
# Distributed under the MIT License, see bottom of file.

import numpy as np
import pyrr
import svgwrite
import random

from typing import NamedTuple, Callable, Sequence

cross = pyrr.vector3.cross
normalize = pyrr.vector3.normalize


class Viewport(NamedTuple):
    minx: float = -0.5
    miny: float = -0.5
    width: float = 1.0
    height: float = 1.0

    @classmethod
    def from_string(cls, string_to_parse):
        args = [float(f) for f in string_to_parse.split()]
        return cls(*args)


class Camera(NamedTuple):
    view_matrix: np.ndarray
    projection_matrix: np.ndarray


class Mesh:
    faces: np.ndarray
    shader: Callable[[int, float], dict] = None
    style: dict = None
    circle_radius: float = 0

    def __init__(self, faces, shader=None, style=None, circle_radius=0):
        self.faces = faces
        self.shader = shader
        self.style = style
        self.circle_radius = circle_radius
        self._bsp_root = None


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

            clip_path = drawing.defs.add(drawing.clipPath())
            clip_min = view.viewport.minx, view.viewport.miny
            clip_size = view.viewport.width, view.viewport.height
            clip_path.add(drawing.rect(clip_min, clip_size))

            for mesh in view.scene.meshes:
                g = self._create_group(drawing, view, mesh)
                g["clip-path"] = clip_path.get_funciri()
                drawing.add(g)

    class _BspNode:
        def __init__(self, face_index, parent):
            self.faces = [face_index]
            self.parent = parent
            self.left = None
            self.right = None
            self.plane = None

    def _create_group(self, drawing, view, mesh):
        faces = mesh.faces
        shader = mesh.shader or (lambda face_index, winding: {})
        default_style = mesh.style or {}
        viewport = view.viewport
        view_matrix = view.camera.view_matrix
        projection_matrix = view.camera.projection_matrix

        # Extend each coordinate to a vec4.
        faces = np.dstack([faces, np.ones(faces.shape[:2])])

        # Sort faces from back to front.
        face_indices = self._sort_back_to_front(mesh, faces, view_matrix)
        faces = faces[face_indices]

        # Apply projection, then apply perspective correction.
        faces = np.dot(faces, np.dot(view_matrix, projection_matrix))
        faces = faces[:, :, :3] / faces[:, :, 3:4]

        # Apply the viewport transform to X and Y.
        faces[:, :, 0] = (1.0 + faces[:, :, 0]) * viewport.width / 2
        faces[:, :, 1] = (1.0 - faces[:, :, 1]) * viewport.height / 2
        faces[:, :, 0] += viewport.minx
        faces[:, :, 1] += viewport.miny

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

    def _sort_back_to_front(self, mesh, faces, view_matrix):
        if faces.shape[1] < 3:
            return np.arange(len(faces))

        # faces = np.dot(faces, view_matrix)
        # z_centroids = np.sum(faces[:, :, 2], axis=1) / faces.shape[1]
        # return np.argsort(z_centroids)

        if mesh._bsp_root is None:
            self._build_bsp(mesh, faces)

        camera_position = pyrr.matrix44.inverse(view_matrix)[3]
        return self._traverse_bsp(mesh, camera_position)

    def _build_bsp(self, mesh, faces):

        # TODO: remove this loop, similar to what we did for windings
        planes = np.zeros((faces.shape[0], 4))
        for face_index, face in enumerate(faces):
            face = face[:, :3]
            p0, p1, p2 = face[[0, 1, 2]]
            normal = cross(p1 - p0, p2 - p0)
            length = pyrr.vector3.length(normal)
            if length > 0:
                normal /= length
            planes[face_index][:3] = normal
            planes[face_index][3] = -np.dot(normal, p0)

        def test_face(face, plane):
            side = 0
            eps = 0.0001
            for vert in face:
                dp = np.dot(vert, plane)
                if dp < -eps:
                    side |= 1
                elif dp > eps:
                    side |= 2
            return side

        # This doesn't really count the number of splits that each
        # candidate would incur, instead it simply tries to find any
        # non-splitting candidate. If it fails, it returns the first
        # candidate.
        def find_least_splitting(descendants):
            for candidate_index, candidate in enumerate(descendants):
                split_count = 0
                plane = planes[candidate]
                for index in descendants:
                    if candidate == index: continue
                    result = test_face(faces[index], plane)
                    if result == 3:
                        split_count += 1
                        break
                if split_count == 0:
                    return descendants.pop(candidate_index)
            print("oh no", len(descendants))
            return descendants.pop(0)

        def recurse(parent, descendants):
            parent.plane = planes[parent.faces[0]]
            if not descendants:
                return

            left_descendants = []
            right_descendants = []
            for face_index in descendants:
                result = test_face(faces[face_index], parent.plane)
                if result & 1: left_descendants.append(face_index)
                if result & 2: right_descendants.append(face_index)
                if result == 0: parent.faces.append(face_index)

            if left_descendants:
                left_child = find_least_splitting(left_descendants)
                parent.left = self._BspNode(left_child, parent)
                recurse(parent.left, left_descendants)

            if right_descendants:
                right_child = find_least_splitting(right_descendants)
                parent.right = self._BspNode(right_child, parent)
                recurse(parent.right, right_descendants)

        descendants = list(range(len(faces)))
        random.Random(4).shuffle(descendants)
        face_index = find_least_splitting(descendants)
        mesh._bsp_root = self._BspNode(face_index, None)
        recurse(mesh._bsp_root, descendants)

    def _traverse_bsp(self, mesh, camera_position):
        ordered = []
        rendered = set()

        def emit_faces(node):
            for index in node.faces:
                if index not in rendered:
                    ordered.append(index)
                    rendered.add(index)

        def recurse(node):
            if not node:
                return
            result = np.dot(node.plane, camera_position)
            if result > 0:
                recurse(node.left)
                emit_faces(node)
                recurse(node.right)
            else:
                recurse(node.right)
                emit_faces(node)
                recurse(node.left)

        recurse(mesh._bsp_root)
        return ordered


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
