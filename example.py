import numpy, svg3d, pyrr, math


def get_octahedron_faces():
    f = math.sqrt(2.0) / 2.0
    verts = numpy.float32(
        [(0, -1, 0), (-f, 0, f), (f, 0, f), (f, 0, -f), (-f, 0, -f), (0, 1, 0)]
    )
    triangles = numpy.int32(
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
    return 15.0 * verts[triangles]


def generate_svg(filename):
    view = pyrr.matrix44.create_look_at(
        eye=[50, 40, 120], target=[0, 0, 0], up=[0, 1, 0]
    )
    projection = pyrr.matrix44.create_perspective_projection(
        fovy=15, aspect=1, near=10, far=200
    )
    camera = svg3d.Camera(view, projection)

    style = dict(
        fill="white",
        fill_opacity="0.75",
        stroke="black",
        stroke_linejoin="round",
        stroke_width="0.005",
    )

    mesh = svg3d.Mesh(get_octahedron_faces(), style=style)
    view = svg3d.View(camera, svg3d.Scene([mesh]))
    svg3d.Engine([view]).render(filename)


generate_svg("octahedron.svg")
