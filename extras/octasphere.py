# This script can generate spheres, rounded cubes, and capsules.
# For more information, see https://prideout.net/blog/octasphere/
# Copyright (c) 2019 Philip Rideout
# Distributed under the MIT License, see bottom of file.

import numpy as np
import pyrr

from math import *

quaternion = pyrr.quaternion

def octasphere(ndivisions: int, radius: float, width=0, height=0, depth=0):
    """Generates a triangle mesh for a sphere, rounded cube, or capsule.

    The ndivisions argument can be used to control the level of detail
    and should be between 0 and 5, inclusive.

    To create a sphere, simply omit the width/height/depth arguments.

    To create a capsule, set one of width/height/depth to a value
    greater than twice the radius. To create a cuboid, set two or more
    of these to a value greater than twice the radius.

    Returns a two-tuple: a numpy array of 3D vertex positions,
    and a numpy array of integer 3-tuples for triangle indices.
    """
    r2 = 2 * radius
    width = max(width, r2)
    height = max(height, r2)
    depth = max(depth, r2)
    n = 2**ndivisions + 1
    num_verts = n * (n + 1) // 2
    verts = np.empty((num_verts, 3))
    j = 0
    for i in range(n):
        theta = pi * 0.5 * i / (n - 1)
        point_a = [0, sin(theta), cos(theta)]
        point_b = [cos(theta), sin(theta), 0]
        num_segments = n - 1 - i
        j = compute_geodesic(verts, j, point_a, point_b, num_segments)
    assert len(verts) == num_verts
    verts = verts * radius

    num_faces = (n - 2) * (n - 1) + n - 1
    faces = np.empty((num_faces, 3), dtype=np.int32)
    f, j0 = 0, 0
    for col_index in range(n-1):
        col_height = n - 1 - col_index
        j1 = j0 + 1
        j2 = j0 + col_height + 1
        j3 = j0 + col_height + 2
        for row in range(col_height - 1):
            faces[f + 0] = [j0 + row, j1 + row, j2 + row]
            faces[f + 1] = [j2 + row, j1 + row, j3 + row]
            f = f + 2
        row = col_height - 1
        faces[f] = [j0 + row, j1 + row, j2 + row]
        f = f + 1
        j0 = j2

    euler_angles = np.float32([
        [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0],
        [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3],
    ]) * pi * 0.5
    quats = (quaternion.create_from_eulers(e) for e in euler_angles)

    offset, combined_verts, combined_faces = 0, [], []
    for quat in quats:
        rotated_verts = [quaternion.apply_to_vector(quat, v) for v in verts]
        rotated_faces = faces + offset
        combined_verts.append(rotated_verts)
        combined_faces.append(rotated_faces)
        offset = offset + len(verts)

    verts = np.vstack(combined_verts)

    tx = (width - r2) / 2
    ty = (height - r2) / 2
    tz = (depth - r2) / 2
    translation = np.float32([tx, ty, tz])

    if np.any(translation):

        translation = np.float32([
            [+1, +1, +1], [+1, +1, -1], [-1, +1, -1], [-1, +1, +1],
            [+1, -1, +1], [-1, -1, +1], [-1, -1, -1], [+1, -1, -1],
        ]) * translation
        for i in range(0, len(verts), num_verts):
            verts[i:i+num_verts] += translation[i // num_verts]

        boundaries = get_boundary_indices(ndivisions)
        assert len(boundaries) == 3
        connectors = []

        def connect(a, b, c, d):
            if np.allclose(verts[a], verts[b]): return
            if np.allclose(verts[b], verts[d]): return
            connectors.append([a, b, c])
            connectors.append([c, d, a])

        if radius > 0:
            # Top half
            for patch in range(4):
                if patch % 2 == 0 and tz == 0: continue
                if patch % 2 == 1 and tx == 0: continue
                next_patch = (patch + 1) % 4
                boundary_a = boundaries[1] + num_verts * patch
                boundary_b = boundaries[0] + num_verts * next_patch
                for i in range(n-1):
                    a = boundary_a[i]
                    b = boundary_b[i]
                    c = boundary_a[i+1]
                    d = boundary_b[i+1]
                    connect(a, b, d, c)
            # Bottom half
            for patch in range(4,8):
                if patch % 2 == 0 and tx == 0: continue
                if patch % 2 == 1 and tz == 0: continue
                next_patch = 4 + (patch + 1) % 4
                boundary_a = boundaries[0] + num_verts * patch
                boundary_b = boundaries[2] + num_verts * next_patch
                for i in range(n-1):
                    a = boundary_a[i]
                    b = boundary_b[i]
                    c = boundary_a[i+1]
                    d = boundary_b[i+1]
                    connect(d, b, a, c)
            # Connect top patch to bottom patch
            if ty > 0:
                for patch in range(4):
                    next_patch = 4 + (4 - patch) % 4
                    boundary_a = boundaries[2] + num_verts * patch
                    boundary_b = boundaries[1] + num_verts * next_patch
                    for i in range(n-1):
                        a = boundary_a[i]
                        b = boundary_b[n-1-i]
                        c = boundary_a[i+1]
                        d = boundary_b[n-1-i-1]
                        connect(a, b, d, c)

        if tx > 0 or ty > 0:
            # Top hole
            a = boundaries[0][-1]
            b = a + num_verts
            c = b + num_verts
            d = c + num_verts
            connect(a, b, c, d)
            # Bottom hole
            a = boundaries[2][0] + num_verts * 4
            b = a + num_verts
            c = b + num_verts
            d = c + num_verts
            connect(a, b, c, d)

        # Side holes
        sides = []
        if ty > 0: sides = [(7,0),(1,2),(3,4),(5,6)]
        for i, j in sides:
            patch_index = i
            patch = patch_index // 2
            next_patch = 4 + (4 - patch) % 4
            boundary_a = boundaries[2] + num_verts * patch
            boundary_b = boundaries[1] + num_verts * next_patch
            if patch_index % 2 == 0:
                a,b = boundary_a[0], boundary_b[n-1]
            else:
                a,b = boundary_a[n-1], boundary_b[0]
            patch_index = j
            patch = patch_index // 2
            next_patch = 4 + (4 - patch) % 4
            boundary_a = boundaries[2] + num_verts * patch
            boundary_b = boundaries[1] + num_verts * next_patch
            if patch_index % 2 == 0:
                c,d = boundary_a[0], boundary_b[n-1]
            else:
                c,d = boundary_a[n-1], boundary_b[0]
            connect(a, b, d, c)

        if radius == 0:
            assert len(connectors) // 2 == 6
            combined_faces = connectors
        else:
            combined_faces.append(connectors)

    return verts, np.vstack(combined_faces)


def compute_geodesic(dst, index, point_a, point_b, num_segments):
    """Given two points on a unit sphere, returns a sequence of surface
    points that lie between them along a geodesic curve."""
    angle_between_endpoints = acos(np.dot(point_a, point_b))
    rotation_axis = np.cross(point_a, point_b)
    dst[index] = point_a
    index = index + 1
    if num_segments == 0:
        return index
    dtheta = angle_between_endpoints / num_segments
    for point_index in range(1, num_segments):
        theta = point_index * dtheta
        q = quaternion.create_from_axis_rotation(rotation_axis, theta)
        dst[index] = quaternion.apply_to_vector(q, point_a)
        index = index + 1
    dst[index] = point_b
    return index + 1


def get_boundary_indices(ndivisions):
    "Generates the list of vertex indices for all three patch edges."
    n = 2**ndivisions + 1
    boundaries = np.empty((3, n), np.int32)
    a, b, c, j0 = 0, 0, 0, 0
    for col_index in range(n-1):
        col_height = n - 1 - col_index
        j1 = j0 + 1
        boundaries[0][a] = j0
        a = a + 1
        for row in range(col_height - 1):
            if col_height == n - 1:
                boundaries[2][c] = j0 + row
                c = c + 1
        row = col_height - 1
        if col_height == n - 1:
            boundaries[2][c] = j0 + row
            c = c + 1
            boundaries[2][c] = j1 + row
            c = c + 1
        boundaries[1][b] = j1 + row
        b = b + 1
        j0 = j0 + col_height + 1
    boundaries[0][a] = j0 + row
    boundaries[1][b] = j0 + row
    return boundaries


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
