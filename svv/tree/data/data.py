import numpy


class TreeMap(dict):
    """
    The TreeMap class defines a mapping between vessels
    in the tree and their upstream and downstream vessels.
    """
    def __new__(cls, *args, **kwargs):
        data = super().__new__(cls, *args, **kwargs)
        return data


class TreeParameters(object):
    def __init__(self):
        """
        The TreeParameters class defines the parameters
        that are used to generate a synthetic vascular
        tree.
        """
        self.kinematic_viscosity = 3.6e-2
        self.fluid_density = 1.06
        self.murray_exponent = 3.0
        self.radius_exponent = 2.0
        self.length_exponent = 1.0
        self.terminal_pressure = 60.0 * 1333.22
        self.root_pressure = 100.0 * 1333.22
        self.terminal_flow = 0.125/2000
        self.root_flow = None
        self.max_nonconvex_count = 100

    def __str__(self):
        output = "Tree Parameters:\n"
        output += "----------------\n"
        output += "Kinematic Viscosity: {}\n".format(self.kinematic_viscosity)
        output += "Fluid Density: {}\n".format(self.fluid_density)
        output += "Murray Exponent: {}\n".format(self.murray_exponent)
        output += "Radius Exponent: {}\n".format(self.radius_exponent)
        output += "Length Exponent: {}\n".format(self.length_exponent)
        output += "Terminal Pressure: {}\n".format(self.terminal_pressure)
        output += "Root Pressure: {}\n".format(self.root_pressure)
        output += "Terminal Flow: {}\n".format(self.terminal_flow)
        output += "Root Flow: {}".format(self.root_flow)
        return output

    def __repr__(self):
        output = "Tree Parameters:\n"
        output += "----------------\n"
        output += "Kinematic Viscosity: {}\n".format(self.kinematic_viscosity)
        output += "Fluid Density: {}\n".format(self.fluid_density)
        output += "Murray Exponent: {}\n".format(self.murray_exponent)
        output += "Radius Exponent: {}\n".format(self.radius_exponent)
        output += "Length Exponent: {}\n".format(self.length_exponent)
        output += "Terminal Pressure: {}\n".format(self.terminal_pressure)
        output += "Root Pressure: {}\n".format(self.root_pressure)
        output += "Terminal Flow: {}\n".format(self.terminal_flow)
        output += "Root Flow: {}".format(self.root_flow)
        return output

    def set(self, parameter, value):
        """
        Set a parameter of the tree.
        """
        if parameter == 'kinematic_viscosity':
            self.kinematic_viscosity = value
        elif parameter == 'fluid_density':
            self.fluid_density = value
        elif parameter == 'murray_exponent':
            self.murray_exponent = value
        elif parameter == 'radius_exponent':
            self.radius_exponent = value
        elif parameter == 'length_exponent':
            self.length_exponent = value
        elif parameter == 'terminal_pressure':
            self.terminal_pressure = value
        elif parameter == 'root_pressure':
            self.root_pressure = value
        elif parameter == 'terminal_flow':
            self.terminal_flow = value
        elif parameter == 'root_flow':
            self.root_flow = value
        else:
            raise ValueError("Invalid parameter: {}.".format(parameter))
        return None


class TreeData(numpy.ndarray):
    """
    TreeData is a subclass of the numpy.ndarray class
    that is used to store synthetic vascular data.
    """
    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            shape = (1, 31)
            data = super().__new__(cls, shape, **kwargs)
            data[:] = numpy.nan
            return data
        elif isinstance(args[0], numpy.ndarray):
            return args[0].view(cls)
        else:
            data = super().__new__(cls, *args, **kwargs)
            data[:] = numpy.nan
            return data

    @classmethod
    def from_array(cls, arr):
        if isinstance(arr, numpy.ndarray):
            return arr.view(cls)
        else:
            return None

    def __array_finalize__(self, obj):
        if obj is None:
            return None

    def __proximal__(self, *args):
        if len(args) == 0:
            return self[:, 0:3].copy()
        else:
            return self[args[0], 0:3].copy()

    def __distal__(self, *args):
        if len(args) == 0:
            return self[:, 3:6].copy()
        else:
            return self[args[0], 3:6].copy()

    def __u_basis__(self, *args):
        if len(args) == 0:
            return self[:, 6:9].copy()
        else:
            return self[args[0], 6:9].copy()

    def __v_basis__(self, *args):
        if len(args) == 0:
            return self[:, 9:12].copy()
        else:
            return self[args[0], 9:12].copy()

    def __w_basis__(self, *args):
        if len(args) == 0:
            return self[:, 12:15].copy()
        else:
            return self[args[0], 12:15].copy()

    def __children__(self, *args):
        if len(args) == 0:
            return self[:, 15:17].copy()
        else:
            return self[args[0], 15:17].copy()

    def __parent__(self, *args):
        if len(args) == 0:
            return self[:, 17].copy()
        else:
            return self[args[0], 17].copy()

    def __proximal_node__(self, *args):
        if len(args) == 0:
            return self[:, 18].copy()
        else:
            return self[args[0], 18].copy()

    def __distal_node__(self, *args):
        if len(args) == 0:
            return self[:, 19].copy()
        else:
            return self[args[0], 19].copy()

    def __length__(self, *args):
        if len(args) == 0:
            return self[:, 20].copy()
        else:
            return self[args[0], 20].copy()

    def __radius__(self, *args):
        if len(args) == 0:
            return self[:, 21].copy()
        else:
            return self[args[0], 21].copy()

    def __flow__(self, *args):
        if len(args) == 0:
            return self[:, 22].copy()
        else:
            return self[args[0], 22].copy()

    def __left_bifurcation__(self, *args):
        if len(args) == 0:
            return self[:, 23].copy()
        else:
            return self[args[0], 23].copy()

    def __right_bifurcation__(self, *args):
        if len(args) == 0:
            return self[:, 24].copy()
        else:
            return self[args[0], 24].copy()

    def __reduced_resistance__(self, *args):
        if len(args) == 0:
            return self[:, 25].copy()
        else:
            return self[args[0], 25].copy()

    def __depth__(self, *args):
        if len(args) == 0:
            return self[:, 26].copy()
        else:
            return self[args[0], 26].copy()

    def __reduced_length__(self, *args):
        if len(args) == 0:
            return self[:, 27].copy()
        else:
            return self[args[0], 27].copy()

    def __radius_scaling__(self, *args):
        if len(args) == 0:
            return self[:, 28].copy()
        else:
            return self[args[0], 28].copy()

    def get(self, label, *args):
        if label == 'proximal':
            return self.__proximal__(*args)
        elif label == 'distal':
            return self.__distal__(*args)
        elif label == 'u_basis':
            return self.__u_basis__(*args)
        elif label == 'v_basis':
            return self.__v_basis__(*args)
        elif label == 'w_basis':
            return self.__w_basis__(*args)
        elif label == 'children':
            return self.__children__(*args)
        elif label == 'parent':
            return self.__parent__(*args)
        elif label == 'proximal_node':
            return self.__proximal_node__(*args)
        elif label == 'distal_node':
            return self.__distal_node__(*args)
        elif label == 'length':
            return self.__length__(*args)
        elif label == 'radius':
            return self.__radius__(*args)
        elif label == 'flow':
            return self.__flow__(*args)
        elif label == 'left_bifurcation':
            return self.__left_bifurcation__(*args)
        elif label == 'right_bifurcation':
            return self.__right_bifurcation__(*args)
        elif label == 'reduced_resistance':
            return self.__reduced_resistance__(*args)
        elif label == 'depth':
            return self.__depth__(*args)
        elif label == 'reduced_length':
            return self.__reduced_length__(*args)
        elif label == 'radius_scaling':
            return self.__radius_scaling__(*args)
        else:
            raise ValueError("Invalid label.")
