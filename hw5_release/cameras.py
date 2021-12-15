from typing import Tuple

import numpy as np


def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
    T = np.eye(4)
    # YOUR CODE HERE
    T[2,3] = d
    T[0,0] = T[2][0] = T[2][2] = -1 / np.sqrt(2)
    T[0,2] = 1 / np.sqrt(2)
    # END YOUR CODE
    assert T.shape == (4, 4)
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)

    # You'll replace this!
    points_transformed = np.zeros((3, N))

    # YOUR CODE HERE
    homo_points = np.concatenate((points, np.ones((1, N))), axis=0)
    homo_points_transformed = np.dot(T, homo_points)
    points_transformed = homo_points_transformed[:3,:] / homo_points_transformed[3,:]
    points_transformed = points_transformed[:3,:]
    # END YOUR CODE

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray: the intersection of the two lines definied by (a0, a1)
                    and (b0, b1).
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == np.float

    # Intersection point between lines
    out = np.zeros(2)

    # YOUR CODE HERE
    slopeA = (a_1[1] - a_0[1]) / (a_1[0] - a_0[0])
    shiftA = a_0[1] - slopeA * a_0[0]
    slopeB = (b_1[1] - b_0[1]) / (b_1[0] - b_0[0])
    shiftB = b_0[1] - slopeB * b_0[0]
    
    x = (shiftB - shiftA) / (slopeA - slopeB)
    y = slopeA * x + shiftA
    out[0], out[1] = x, y
    # END YOUR CODE

    assert out.shape == (2,)
    assert out.dtype == np.float

    return out


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)

    # YOUR CODE HERE
    point1 = np.array([0, (v0[0]-v1[0])*(-v2[0])/(v1[1]-v0[1]) + v2[1]])
    point2 = np.array([1, (v0[0]-v1[0])*(1-v2[0])/(v1[1]-v0[1]) + v2[1]])
    M = intersection_from_lines(point1, point2, v0, v1)
    
    point3 = np.array([0, (v1[0]-v2[0])*(-v0[0])/(v2[1]-v1[1]) + v0[1]])
    point4 = np.array([1, (v1[0]-v2[0])*(1-v0[0])/(v2[1]-v1[1]) + v0[1]])
    N = intersection_from_lines(point3, point4, v1, v2)
    
    optical_center = intersection_from_lines(M, v2, N, v0)
    # END YOUR CODE

    assert optical_center.shape == (2,)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    f = None

    # YOUR CODE HERE
    point1 = np.array([0, (v0[0]-v1[0])*(-optical_center[0])/(v1[1]-v0[1]) + optical_center[1]])
    point2 = np.array([1, (v0[0]-v1[0])*(1-optical_center[0])/(v1[1]-v0[1]) + optical_center[1]])
    M = intersection_from_lines(point1, point2, v0, v1)
    f = np.sqrt(np.linalg.norm(M - v0) * np.linalg.norm(M - v1) - np.linalg.norm(M - optical_center)**2)
    # END YOUR CODE

    return float(f)


def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    f_mm = None

    # YOUR CODE HERE
    f_mm = f * sensor_diagonal_mm / image_diagonal_pixels
    # END YOUR CODE

    return f_mm
