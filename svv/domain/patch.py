import numpy as np
from .kernel.kernel import Kernel
from .solver.solver import Solver
from time import perf_counter_ns


class Patch:
    """
    Patch class decomposes a domain into a set of interpolation sub-problems which
    can be solved independently and then blended together to form the final
    domain interpolation function.
    """
    def __init__(self, lam=0):
        self.points = None
        self.normals = None
        self.kernel = None
        self.solver = None
        self.constants = None
        self.rbf_degree = 3
        self.lam = lam

    def set_data(self, *args):
        """
        Set the data for the domain.
        """
        if len(args) > 1:
            self.points = args[0]
            self.normals = args[1]
            self.kernel = Kernel(self.points, lam=self.lam)
            self.kernel.set_initial_values(self.normals)
        elif len(args) > 0:
            self.points = args[0]
            self.kernel = Kernel(self.points, lam=self.lam)
            self.kernel.set_initial_values()
        else:
            print("Error: No data provided.")
        return None

    def solve(self, method="L-BFGS-B", precision=9):
        """
        Solve the interpolation problem for the patch object

        Parameters:
        -----------
        method : str
            The method to use for the solver. Default is L-BFGS-B.
            (Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bounded)

        precision : int
            The number of decimal places to round the constants to.
            Default is 9.

        Returns:
        --------
        None
        """
        self.solver = Solver(self.kernel)
        self.solver.set_solver(method=method)
        if isinstance(self.normals, type(None)):
            self.solver.solve()
        else:
            self.solver.solve(skip=True)
        self.constants = np.round(self.solver.get_constants(), decimals=precision)
        return None

    def build(self):
        """
        Build the interpolation function for the patch object
        :return:
        """
        a = np.array(self.constants[:self.kernel.n])
        b = np.array(self.constants[self.kernel.n:self.kernel.n * (self.kernel.d + 1)].reshape(self.kernel.n, self.kernel.d))
        c = np.array(self.constants[self.kernel.n * (self.kernel.d + 1):self.kernel.n * (self.kernel.d + 1) + self.kernel.d])
        d = np.array(self.constants[-1])
        a = a.reshape(a.shape + (1,) * self.kernel.d)
        b = b.reshape(tuple([b.shape[0]]) + (1,) * self.kernel.d + tuple([b.shape[1]]))
        c = c.reshape((1,) * self.kernel.d + c.shape)
        pts = np.array(self.points.reshape(tuple([self.points.shape[0]]) + (1,) * self.kernel.d + tuple([self.points.shape[1]])))
        def f(x, a_=a, b_=b, c_=c, d_=d, points=pts, show=False):
            """
            Interpolation function for a patch of the domain.

            Parameters:
            -----------
            x : array_like
                The point(s) at which to evaluate the interpolation function. Given as an
                ndarray of shape (..., d), where d is the dimension of the domain.

            a_ : array_like
                The coefficients for the Duchon interpolation function.

            b_ : array_like
                The coefficients for the gradient of the Duchon interpolation function.

            c_ : array_like
                The coefficients for the Hessian of the Duchon interpolation function.

            d_ : float
                The constant term for the Duchon interpolation function.

            Returns:
            --------
            value : array_like
                The value of the interpolation function at the given point(s). Given as an
                ndarray of shape (..., 1).
            """
            diff = x - points
            diff_2 = np.sum(np.square(diff), axis=-1)
            if show:
                print("Diff: ", diff)
                print("Diff^2: ", diff_2)
            #a_value = np.sum(a_ * np.sum(np.square(diff), axis=-1) ** (self.rbf_degree / 2), axis=0)
            # A Value
            value = np.sum(a_ * np.power(diff_2, (self.rbf_degree / 2)), axis=0)
            if show:
                print("A Value: ", value)
            # B Value
            value += np.sum(self.rbf_degree * np.sum((-b_) * diff, axis=-1) * np.power(diff_2, (
                        self.rbf_degree / 2 - 1)), axis=0)
            if show:
                print("A + B Value: ", value)
            # C Value
            value += np.sum(c_ * x, axis=-1)
            if show:
                print("A + B + C Value: ", value)
            # D Value
            value += d_
            if show:
                print("A + B + C + D Value: ", value)
            return value
        f.points = self.points
        f.pts = pts
        if self.normals is not None:
            f.normals = self.normals
        else:
            f.normals = self.solver.get_normals()
        f.dimensions = self.kernel.d
        f.min = np.min(self.points, axis=0)
        f.max = np.max(self.points, axis=0)
        f.centroid = np.mean(self.points, axis=0)
        f.first = self.points[0, :]
        f.a = a
        f.b = b
        f.c = c
        f.d = d
        return f
