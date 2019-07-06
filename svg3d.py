import numpy as np
import pyrr
import svgwrite

from typing import NamedTuple, Callable

class Viewport(NamedTuple):
    minx: float = -.5
    miny: float = -.5
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
    shader: Callable[[int, float], dict]
    style: dict = {}

class Scene:

    def __init__(self, mesh=None):
        self.meshes = []
        if mesh:
            self.meshes.append(mesh)

    def add_mesh(self, mesh: Mesh):
        self.meshes.append(mesh)

class View(NamedTuple):
    camera: Camera
    scene: Scene
    viewport: Viewport = Viewport()

class Engine:

    def __init__(self, views):
        self.views = views

    def render(self, filename, size=(512,512), viewBox='-0.5 -0.5 1.0 1.0'):
        drawing = svgwrite.Drawing(filename, size, viewBox=viewBox)
        for view in self.views:
            projection = np.dot(view.camera.view, view.camera.projection)
            for mesh in view.scene.meshes:
                drawing.add(self._create_group(drawing, projection, view.viewport, mesh))
        drawing.save()

    def _create_group(self, drawing, projection, viewport, mesh):
        faces = mesh.faces

        # Extend each point to a vec4, then multiply by the MVP.
        ones = np.ones(faces.shape[:2] + (1,))
        faces = np.dstack([faces, ones])
        faces = np.dot(faces, projection)

        # Divide X Y Z by W.
        faces = faces[:,:,:3] / faces[:,:,3:4]

        # Apply viewport transform.
        faces[:,:,0:2] = faces[:,:,0:2] + 1
        faces[:,:,0:2] = faces[:,:,0:2] * viewport.dims() / 2
        faces[:,:,0:2] = faces[:,:,0:2] + viewport.min()

        # Sort faces from back to front.
        z_centroids = np.flip(np.sum(faces[:,:,2], axis=1))
        face_indices = np.argsort(z_centroids)
        faces = faces[face_indices]

        # Compute the winding direction of each polygon, determine its
        # style, and add it to the group. If the returned style is None,
        # cull away the polygon.
        group = drawing.g(**mesh.style)
        face_index = 0
        for face in faces:
            p0, p1, p2 = face[0], face[1], face[2]
            winding = -pyrr.vector3.cross(p2 - p0, p1 - p0)[2]
            style = mesh.shader(face_indices[face_index], winding)
            if style:
                group.add(drawing.polygon(face[:,0:2], **style))
            face_index = face_index + 1

        return group
