import numpy as np

__all__ = ["HexGrid", "FiniteHexGrid"]


class HexGrid:
    """
    A way of representing points that lie on a hexagonal grid.

    This class uses arrays to represent a collection of points on a hexagonal grid.
    This follows the convention that the first axis is the number of points, and the second
    axis is the number of dimensions.
    """

    def __init__(self, grid_spacing, rotation, origin):
        # if rotation < -2 * np.pi or rotation > 2 * np.pi:
        #     raise RuntimeWarning("Are you sure rotation is in radians?")

        # promote origin vector to 2d if needed
        origin = np.array(origin) if not isinstance(origin, np.ndarray) else origin
        if origin.ndim == 1:
            origin = origin[None, :]

        if grid_spacing < 0:
            raise ValueError("grid_spacing must be > 0")

        self._grid_spacing = grid_spacing
        self._rotation = rotation
        self._origin = origin

        self._basis_matrix = np.array(
            [[1.0, np.cos(np.pi / 3)], [0.0, np.sin(np.pi / 3)]]
        )

    def update(self, theta):
        """
        Update the grid parameters, equivalent to a set all function

        Args:
        -----
        theta : list
            [grid_spacing, rotation, origin_x, origin_y]

        Returns:
        --------
        None
        """
        self._grid_spacing = theta[0]
        self._rotation = theta[1]
        self._origin = np.array(theta[2:4])[None, :]

    def basis_to_xy(self, basis_vec):
        """
        Convert basis vectors to xy points

        Args:
        -----
        basis_vec : np.ndarray
            (N, 2) The basis vectors to convert

        Returns:
        --------
        np.ndarray
            (N, 2) The xy points
        """
        if basis_vec.ndim == 1:
            raise ValueError("basis_vec must be 2D")
        if basis_vec.shape[1] != 2:
            raise ValueError("basis_vec must have shape (N, 2)")

        xy = (self._grid_spacing * basis_vec) @ (
            self._rot_matrix @ self._basis_matrix
        ).T + self._origin

        return xy

    def xy_to_basis(self, xy, is_round=False):
        """
        Convert xy points to basis vectors

        Args:
        -----
        xy : np.ndarray
            (N, 2) The xy points to convert
        is_round : bool
            Whether to round the basis vectors to the nearest integer

        Returns:
        --------
        np.ndarray
            (N, 2) The basis vectors
        """
        if xy.ndim == 1:
            raise ValueError("xy must be 2D")
        if xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")

        basis_vec = (
            (xy - self._origin) @ np.linalg.inv(self._rot_matrix @ self._basis_matrix).T
        ) / self._grid_spacing
        if is_round:
            basis_vec = np.round(basis_vec)
        return basis_vec

    def __str__(self) -> str:
        return f"Hexgrid with Grid spacing: {self._grid_spacing}, Rotation: {self._rotation}, Origin: {self._origin}"

    @property
    def _rot_matrix(self):
        return np.array(
            [
                [np.cos(self._rotation), -np.sin(self._rotation)],
                [np.sin(self._rotation), np.cos(self._rotation)],
            ]
        )


class FiniteHexGrid(HexGrid):
    """
    A finite hexagonal grid, with a given radius and rotation.

    This class uses arrays to represent a collection of points on a hexagonal grid.
    This follows the convention that the first axis is the number of points, and the second
    axis is the number of dimensions.

    Args:
    -----
    grid_spacing : float
        The distance between the centres of the hexagons
    rotation : float
        The rotation of the grid in radians
    origin : np.ndarray
        (2,) The origin of the grid
    hexagonal_radius : int
        The radius of the grid in hexagons e.g. 1 means 7 points, 2 means 19 points etc.
    """

    def __init__(
        self,
        grid_spacing,
        rotation,
        origin,
        hexagonal_radius,
        include_corners=True,
    ):
        super().__init__(grid_spacing, rotation, origin)

        # validate hexagonal_radius
        if hexagonal_radius < 1:
            raise ValueError("hexagonal_radius must be > 0")
        if not isinstance(hexagonal_radius, int):
            raise ValueError("hexagonal_radius must be an integer")

        self._hexagonal_radius = hexagonal_radius

        self._basis_points_full_grid = self._generate_basis_points_up_to_r(
            hexagonal_radius, include_corners
        )

    def __str__(self) -> str:
        return super().__str__() + f"Hexagonal radius: {self._hexagonal_radius}"

    def _generate_basis_points_up_to_r(self, r, include_corners=True):
        if include_corners:
            eps = +0.01
        else:
            eps = -0.01

        u = np.arange(-r, r + 1)

        uu, vv = np.meshgrid(u, u)

        points = []
        for u, v in zip(uu.flatten(), vv.flatten()):
            basis_vec = np.array([[u], [v]])
            xy = self._basis_matrix @ (basis_vec)

            if np.linalg.norm(xy) < r + eps:
                points.append(basis_vec)

        basis_vecs = np.array(points).squeeze()

        # now do the weird reordering trick
        hg = HexGrid(1.0, 0.0, np.array([0.0, 0.0]))

        def order_metric(basis_point):
            xy = hg.basis_to_xy(basis_point)
            r = np.ceil(np.linalg.norm(xy) - 0.01)
            phi = np.arctan2(xy[0, 1], xy[0, 0])
            phi = np.mod(phi, 2 * np.pi)
            return r + phi / (2 * np.pi)

        ordering_vector = np.array(
            [order_metric(vec[None, :]) for vec in basis_vecs]
        ).flatten()

        return basis_vecs[ordering_vector.argsort()]

    def all_xy_points(self):
        """
        Get all the xy points in the grid

        Returns:
        --------
        np.ndarray
            The xy points in the grid
        """
        return self.basis_to_xy(self._basis_points_full_grid)

    @staticmethod
    def hexgrid_radius_to_n_points(radius, include_corners=True):
        """
        Calculate the number of points in a hexagonal grid of a given radius

        Args:
        -----
        radius : int
            The radius of the grid in hexagons e.g. 1 means 7 points, 2 means 19 points etc.
        include_corners : bool
            Whether to include the corners of the grid

        Returns:
        --------
        int
            The number of points in the grid
        """
        if not isinstance(radius, int):
            raise ValueError("radius must be an integer")
        if radius < 1:
            raise ValueError("radius must be > 0")
        n_th_hex_number = 1 + 6 * (radius * (radius + 1)) // 2
        if not include_corners:
            return n_th_hex_number - 6
        return n_th_hex_number

    @staticmethod
    def hexgrid_n_points_to_radius(n_points, include_corners=True):
        """
        Calculate the radius of a hexagonal grid with a given number of points

        Args:
        -----
        n_points : int
            The number of points in the grid
        include_corners : bool
            Whether to include the corners of the grid

        Returns:
        --------
        int
            The radius of the grid in hexagons e.g. 1 means 7 points, 2 means 19 points etc.
        """
        if not isinstance(n_points, int):
            raise ValueError("n_points must be an integer")
        if n_points < 1:
            raise ValueError("n_points must be > 0")
        if not include_corners:
            n_points += 6
        radius = (3 + np.sqrt(12 * n_points - 3)) / 6 - 1
        # check if integer is returned
        if radius != int(radius):
            raise ValueError("n_points does not correspond to a hexagonal grid")

        return radius
