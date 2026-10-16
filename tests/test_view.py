import re

import numpy as np
import numpy.testing as npt
import pyrr
import pytest
import rowan
from conftest import filter_invalid_arrays
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from svg3d import (
    Engine,
    Mesh,
    View,
    get_lookat_matrix,
    get_orthographic_matrix,
    get_projection_matrix,
)

MIN_VALID_DISTANCE = 1e-12
PRECISION = 1e-12  # Small deviations are OK given the rigor of hypothesis testing


@pytest.mark.filterwarnings("ignore: invalid value encountered in divide")
@pytest.mark.parametrize(
    "vec_up", [(0, 1, 0), (1, 0, 0), (0, 0, 1), (123.0, -45, 0.0001)]
)
@given(
    pos=arrays(np.float64, (2, 3), elements=floats(-100, 100)).filter(
        lambda x: (
            (np.linalg.norm(x[0] - x[1]) > MIN_VALID_DISTANCE)
            and filter_invalid_arrays(x, PRECISION)
        )
    ),
)
def test_get_lookat_matrix(pos, vec_up):
    """
    Extensive configuration is required for this test to remove extreme degenerate
    hypothesis cases. For all 'reasonable' inputs, the results are equal to machine
    precision.
    """
    pos_object, pos_camera = pos

    npt.assert_allclose(
        get_lookat_matrix(pos_object, pos_camera, vec_up),
        pyrr.matrix44.create_look_at(eye=pos_camera, target=pos_object, up=vec_up),
        atol=PRECISION,
    )


@given(
    z_near=floats(MIN_VALID_DISTANCE, 100, exclude_min=True),
    z_far_distance=floats(MIN_VALID_DISTANCE, 100, exclude_min=True),
    fov_y=floats(MIN_VALID_DISTANCE, 180, exclude_min=True, exclude_max=True),
    aspect=floats(1 / 100, 100),
)
def test_get_projection_matrix(z_near, z_far_distance, fov_y, aspect):
    z_far = z_near + z_far_distance
    npt.assert_allclose(
        get_projection_matrix(z_near, z_far, fov_y, aspect),
        pyrr.matrix44.create_perspective_projection_matrix(
            fov_y, aspect, z_near, z_far
        ),
        atol=PRECISION,
    )


def test_isometric_view():
    isometric_view_matrix = np.eye(4)
    isometric_view_matrix[:3, :3] = (
        rowan.to_matrix(rowan.from_axis_angle([0, 1, 0], np.pi / 4))  # y-up convention
        @ rowan.to_matrix(rowan.from_axis_angle([1, 0, 0], -np.atan(1 / np.sqrt(2))))
    )
    isometric_view_matrix[-1, 2] = -100

    npt.assert_allclose(
        View.ISOMETRIC_VIEW_MATRIX, isometric_view_matrix, atol=PRECISION
    )


@given(
    z_near=floats(MIN_VALID_DISTANCE, 100, exclude_min=True),
    z_far_distance=floats(MIN_VALID_DISTANCE, 100, exclude_min=True),
    half_width=floats(MIN_VALID_DISTANCE, 100, exclude_min=True),
    half_height=floats(MIN_VALID_DISTANCE, 100, exclude_min=True),
)
def test_get_orthographic_matrix(z_near, z_far_distance, half_width, half_height):
    z_far = z_near + z_far_distance
    npt.assert_allclose(
        get_orthographic_matrix(half_width, half_height, z_near, z_far),
        pyrr.matrix44.create_orthogonal_projection_matrix(
            -half_width, half_width, -half_height, half_height, z_near, z_far
        ),
        atol=PRECISION,
    )


def _projected_axes(view):
    """Project the unit world axes through a view; return 2D image vectors."""
    axes = np.concatenate([np.eye(3), np.ones((3, 1))], axis=1)
    projected = axes @ (view.look_at @ view.projection)
    return projected[:, :2] / projected[:, 3:4]


@pytest.mark.parametrize(
    "view_name, view_direction",
    [
        ("isometric", (1, 1, 1)),
        ("dimetric", (8, 8, 21)),
        ("trimetric", (2, 1, 3)),
    ],
)
def test_axonometric_foreshortening_is_exact(view_name, view_direction):
    """Projected axis lengths must match the closed form sqrt(1 - n_i^2) exactly."""
    n = np.array(view_direction, dtype=np.float64)
    n /= np.linalg.norm(n)
    view = getattr(View, view_name)([])
    # The projection scales the image by 1 / half_width; the foreshortening
    # relative to that common scale must match the closed form sqrt(1 - n_i^2).
    expected = np.sqrt(1 - n**2) * view.projection[0, 0]

    got = np.linalg.norm(_projected_axes(view), axis=1)
    npt.assert_allclose(got, expected, atol=1e-12)


def test_isometric_axes_are_120_degrees_apart():
    axes = _projected_axes(View.isometric([]))
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        cos_theta = np.dot(axes[i], axes[j]) / np.prod(
            np.linalg.norm([axes[i], axes[j]], axis=1)
        )
        npt.assert_allclose(np.degrees(np.arccos(cos_theta)), 120.0, atol=1e-9)


def test_isometric_cube_silhouette_is_a_regular_hexagon(tmp_path):
    """An exactly isometric view of a cube must yield a regular hexagon outline.

    All three axes foreshorten equally to sqrt(2/3), so the silhouette of a unit
    cube is a regular hexagon whose circumradius is the projected distance of the
    "equator" corners from the view axis, sqrt(2/3) * (0.5 / half_width) in SVG
    units, and the near and far cube corners project onto its center.
    """
    mesh = Mesh.example_mesh()
    out = tmp_path / "hexagon.svg"
    Engine([View.isometric([mesh])]).render(str(out))

    points = {
        tuple(round(float(c), 12) for c in pair.split(","))
        for attr in re.findall(r'points="([^"]+)"', out.read_text())
        for pair in attr.split()
    }
    pts = np.array(sorted(points))
    # 8 cube corners, with the near and far corners coinciding at the centre.
    assert len(pts) == 7

    centre = pts.mean(axis=0)
    radii = np.linalg.norm(pts - centre, axis=1)
    outer = radii > radii.max() / 2
    assert outer.sum() == 6

    # Regular hexagon: all six circumradii should equal the closed form.
    half_width = 100.0 * np.tan(np.radians(1.0) / 2)
    expected = 0.5 * np.sqrt(2 / 3) / half_width
    npt.assert_allclose(radii[outer], expected, atol=1e-9)
    # The near and far corners coincide exactly at the hexagon's center.
    assert radii[~outer][0] < 1e-9

    # A regular hexagon's sides equal its circumradius.
    xy = pts[outer] - centre
    ordered = pts[outer][np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))]
    sides = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    npt.assert_allclose(sides, expected, atol=1e-9)


def test_dimetric_two_axes_equal_and_trimetric_none():
    dim = np.linalg.norm(_projected_axes(View.dimetric([])), axis=1)
    npt.assert_allclose(dim[0], dim[1], atol=1e-12)
    assert not np.isclose(dim[0], dim[2], rtol=0, atol=1e-9)

    tri = np.linalg.norm(_projected_axes(View.trimetric([])), axis=1)
    assert len({round(x, 9) for x in tri}) == 3


def test_axonometric_projection_is_parallel():
    """Image of an edge must not depend on its depth along the view direction."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    edge = np.array([[0.5, 0, 0], [-0.5, 0, 0], [-0.5, 0, 0]])

    view = View.isometric([])
    projection = view.look_at @ view.projection
    unit_x_image = _projected_axes(view)[0]
    for depth in (0.0, 1.0, 5.0, 50.0):
        points = np.concatenate([edge + depth * n, np.ones((3, 1))], axis=1)
        image = points @ projection
        image = image[:, :2] / image[:, 3:4]
        # The image of the unit x axis is identical at every depth.
        npt.assert_allclose(image[0] - image[1], unit_x_image, atol=1e-12)


@pytest.mark.parametrize(
    "fov, distance", [(1.0, 100.0), (1.0, 50.0), (90.0, 100.0), (30.0, 250.0)]
)
def test_parallel_projection_window_follows_fov_and_distance(fov, distance):
    """Rendered scale must be exactly 1 / (distance * tan(fov / 2)) per NDC unit."""
    view = View.isometric([], fov=fov, distance=distance)
    half_width = distance * np.tan(np.radians(fov) / 2)
    npt.assert_allclose(view.projection[0, 0], 1.0 / half_width, rtol=1e-12)
    npt.assert_allclose(view.projection[1, 1], 1.0 / half_width, rtol=1e-12)


def test_isometric_does_not_mutate_class_state():
    """Creating views must never modify the shared class attribute."""

    class _Scene:
        pass

    v1 = View.isometric([_Scene()], distance=50.0)
    assert v1.look_at[-1, 2] == -50.0
    assert View.ISOMETRIC_VIEW_MATRIX[-1, 2] == -100.0  # untouched

    v2 = View.isometric([_Scene()], distance=200.0)
    assert v1.look_at[-1, 2] == -50.0  # first view unchanged by the second
    assert v2.look_at[-1, 2] == -200.0
    assert View.ISOMETRIC_VIEW_MATRIX[-1, 2] == -100.0


@pytest.mark.parametrize(
    "fov, distance", [(1.0, 100.0), (1.0, 50.0), (90.0, 100.0), (30.0, 250.0)]
)
def test_parallel_projection_window_follows_fov_and_distance(fov, distance):
    """Rendered scale must be exactly 1 / (distance * tan(fov / 2)) per NDC unit."""
    view = View.isometric([], fov=fov, distance=distance)
    half_width = distance * np.tan(np.radians(fov) / 2)
    npt.assert_allclose(view.projection[0, 0], 1.0 / half_width, rtol=1e-12)
    npt.assert_allclose(view.projection[1, 1], 1.0 / half_width, rtol=1e-12)


@pytest.mark.parametrize("view_name", ["isometric", "dimetric", "trimetric"])
@pytest.mark.parametrize("distance", [100.0, 1000.0])
def test_view_volume_tracks_distance(view_name, distance):
    """The object, and a margin around it, must stay inside the view volume."""
    view = getattr(View, view_name)([], distance=distance)
    inv_look_at = np.linalg.inv(view.look_at)
    # Camera basis directions expressed in world coordinates.
    into_screen = (np.array([0.0, 0.0, -1.0, 0.0]) @ inv_look_at)[:3]
    camera_right = (np.array([1.0, 0.0, 0.0, 0.0]) @ inv_look_at)[:3]
    camera_up = (np.array([0.0, 1.0, 0.0, 0.0]) @ inv_look_at)[:3]
    # Offset half-way towards the edge of the window, so that x and y (not just
    # the depth) are genuinely exercised by the containment check.
    half_width = distance * np.tan(np.radians(1.0) / 2)
    offset = 0.5 * half_width * camera_right + 0.5 * half_width * camera_up

    ndc_z = []
    for depth in (0.5 * distance, distance, 1.5 * distance):
        point = np.append((depth - distance) * into_screen + offset, 1.0)
        ndc = point @ (view.look_at @ view.projection)
        assert np.all(np.abs(ndc[:3]) < 1.0), f"depth {depth} is outside the volume"
        ndc_z.append(ndc[2])
    # Farther points must map to larger NDC z (catches reversed near/far planes).
    assert ndc_z[0] < ndc_z[1] < ndc_z[2]


@pytest.mark.parametrize("view_name", ["isometric", "dimetric", "trimetric"])
def test_named_views_place_object_at_documented_distance(view_name):
    """The object origin must sit exactly ``distance`` in front of the camera."""
    for distance in (1.0, 100.0, 250.0):
        view = getattr(View, view_name)([], distance=distance)
        origin = np.array([0.0, 0.0, 0.0, 1.0]) @ view.look_at
        npt.assert_allclose(origin[2], -distance, atol=1e-9)


@pytest.mark.parametrize("view_name", ["isometric", "dimetric", "trimetric"])
def test_faces_are_drawn_back_to_front(view_name, tmp_path):
    """Painter's algorithm: the farther face must be drawn first."""
    base = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]])
    view = getattr(View, view_name)([])
    # World direction pointing into the screen (away from the camera).
    into_screen = (np.array([0.0, 0.0, -1.0, 0.0]) @ np.linalg.inv(view.look_at))[:3]
    far, near = base + 0.4 * into_screen, base - 0.4 * into_screen
    mesh = Mesh(
        faces=[far, near],
        shader=lambda index, mesh: {"fill": "#ff0000" if index == 0 else "#0000ff"},
    )
    out = tmp_path / "back_to_front.svg"
    Engine([getattr(View, view_name)([mesh])]).render(str(out))
    svg = out.read_text()
    assert svg.index("#ff0000") < svg.index("#0000ff")


def test_faces_of_mixed_sizes_sort_by_centroid_depth(tmp_path):
    """Short faces must not be ordered by their (padded, repeated) first vertex."""
    triangle = np.array([[-1.2, 0.0, -8.0], [-1.0, 0.2, -1.5], [-1.4, 0.2, -1.5]])
    quad = np.array(
        [[0.0, -0.2, -4.7], [0.4, -0.2, -4.7], [0.4, 0.2, -4.7], [0.0, 0.2, -4.7]]
    )
    view = View.from_look_at_and_projection(
        look_at=np.eye(4),
        projection=get_orthographic_matrix(2.0, 2.0, 1.0, 10.0),
        scene=[],
    )
    mesh = Mesh(
        faces=[triangle, quad],
        shader=lambda index, mesh: {"fill": "#ff0000" if index == 0 else "#0000ff"},
    )
    view.scene = [mesh]
    out = tmp_path / "mixed_sizes.svg"
    Engine([view]).render(str(out))
    svg = out.read_text()
    assert svg.index("#0000ff") < svg.index("#ff0000")


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"distance": 0.0},
        {"fov": 0.0},
        {"fov": 180.0},
        # Large enough that 2 * distance overflows to inf
        {"distance": 9.0e307},
    ],
)
def test_named_views_reject_degenerate_parameters(bad_kwargs):
    with pytest.raises(ValueError):
        View.isometric([], **bad_kwargs)


@pytest.mark.parametrize(
    "invalid",
    [
        {"half_width": 0.0},
        {"z_near": 0.0},
        {"z_near": 2.0, "z_far": 1.0},
        {"half_width": np.nan},
        {"z_far": np.inf},
    ],
)
def test_get_orthographic_matrix_rejects_invalid_volumes(invalid):
    valid = {"half_width": 1.0, "half_height": 1.0, "z_near": 1.0, "z_far": 200.0}
    with pytest.raises(ValueError):
        get_orthographic_matrix(**(valid | invalid))
