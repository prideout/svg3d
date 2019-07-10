#!/usr/bin/env python3

import numpy as np
import svg3d
import pygltflib
import pyrr

create_view_matrix = pyrr.matrix44.create_look_at
create_proj_matrix = pyrr.matrix44.create_perspective_projection


def create_meshes_from_glb(filename, scale, style, mesh_index=0):
    glb = pygltflib.GLTF2().load(filename)
    meshes = []
    for prim in glb.meshes[mesh_index].primitives:

        # Indices
        accessor = glb.accessors[prim.indices]
        assert accessor.type == "SCALAR"
        assert not accessor.sparse
        assert accessor.componentType == pygltflib.UNSIGNED_SHORT
        nindices = accessor.count
        bv = glb.bufferViews[accessor.bufferView]
        data = glb._glb_data[bv.byteOffset : bv.byteOffset + bv.byteLength]
        triangles = np.frombuffer(data, dtype=np.uint16)
        triangles = np.reshape(triangles, (-1, 3))
        assert nindices == len(triangles) * 3

        # Vertices
        accessor = glb.accessors[prim.attributes.POSITION]
        assert accessor.type == "VEC3"
        assert not accessor.sparse
        assert accessor.componentType == pygltflib.FLOAT
        nvertices = accessor.count
        bv = glb.bufferViews[accessor.bufferView]
        data = glb._glb_data[bv.byteOffset : bv.byteOffset + bv.byteLength]
        vertices = np.frombuffer(data, dtype=np.float32)
        vertices = np.reshape(vertices, (-1, 3))
        assert nvertices == len(vertices)

        faces = scale * vertices[triangles]
        cull = lambda face_index, winding: None if winding < 0 else {}
        meshes.append(svg3d.Mesh(faces, cull, style=style))

    return meshes


def generate_svg_from_meshes(filename, meshes):
    view = create_view_matrix(eye=[1, -1, -2], target=[0, 0.3, 0], up=[0, 1, 0])
    projection = create_proj_matrix(fovy=15, aspect=1, near=10, far=100)
    camera = svg3d.Camera(view, projection)
    view = svg3d.View(camera, svg3d.Scene(meshes))
    svg3d.Engine([view]).render(filename, style="width: 100%; height: 100%;")


style = dict(
    fill="#e0efe7",
    fill_opacity="0.75",
    stroke="black",
    stroke_linejoin="round",
    stroke_width="0.002",
)

meshes = create_meshes_from_glb("Avocado.glb", 10.0, style)
generate_svg_from_meshes("avocado.svg", meshes)
