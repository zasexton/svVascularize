import sympy as sp
from sympy.interactive import printing
printing.init_printing(use_latex=True)


class BaseUnit(object):
    def __init__(self, value, base, unit=None):
        """
        The BaseUnit class defines the base units that are
        used to generate a synthetic vascular tree.
        """
        valid_bases = ['length', 'time', 'mass']
        base_dimensions = {'length': ['L'],
                           'time': ['T'],
                           'mass': ['M']}
        default_units = {'length': 'cm',
                         'time': 's',
                         'mass': 'g'}
        if base not in valid_bases:
            raise ValueError("Invalid base units: {}.".format(base))
        else:
            if unit is None:
                unit = default_units[base]
        self.dimensions = [base_dimensions[base], []]
        self.unit = sp.Symbol(unit)
        self.base = base
        self.value = value

    def __str__(self):
        return str(self.value) + ' ' + str(self.unit)

    def __mul__(self, other):
        if isinstance(other, BaseUnit):
            if self.base == other.base:
                conversion = ConvertBaseUnit(self, other)
                new_value = conversion.factor
            return self.unit * other.unit

    def __truediv__(self, other):
        if isinstance(other, BaseUnit):
            return self.unit / other.unit

    def __pow__(self, other):
        if isinstance(other, int):
            return self.unit ** other


class BaseUnitSystem(object):
    def __init__(self, **kwargs):
        """
        The BaseUnits class defines the base units that are
        used to generate a synthetic vascular tree.
        """
        length = kwargs.get('length', 'cm')
        time = kwargs.get('time', 's')
        mass = kwargs.get('mass', 'g')
        self.length = BaseUnit('length', length)
        self.time = BaseUnit('time', time)
        self.mass = BaseUnit('mass', mass)

    def __str__(self):
        output = "Base Units:\n"
        output += "------------\n"
        output += "Length: {}\n".format(self.length.unit)
        output += "  Time: {}\n".format(self.time.unit)
        output += "  Mass: {}\n".format(self.mass.unit)
        output += "------------"
        return output


class DerivedUnit(object):
    def __init__(self, value, unit, dimensions):
        """
        The DerivedUnit class defines the derived units that are
        used to generate a synthetic vascular tree.
        """
        self.value = value
        self.unit = unit
        self.dimensions = dimensions

    def __str__(self):
        return str(self.value) + ' ' + str(self.unit)

    def __mul__(self, other):
        if isinstance(other, DerivedUnit):
            return self.unit * other.unit

class NamedUnit(object):
    def __init__(self, derived, base_unit_system, **kwargs):
        """
        The DerivedUnit class defines the derived units that are
        used to generate a synthetic vascular tree.
        """
        self.base_unit_system = base_unit_system
        valid_derived_types = {'pressure': [['M'], ['L', 'T', 'T']],
                               'density': [['M'], ['L', 'L', 'L']],
                               'volumetric_flow': [['L', 'L', 'L'], ['T']],
                               'mass_flow': [['M'], ['T']],
                               'area': [['L', 'L']],
                               'volume': [['L', 'L', 'L']],
                               'velocity': [['L'], ['T']],
                               'acceleration': [['L'], ['T', 'T']],
                               'kinematic_viscosity': [['L', 'L'], ['T']],
                               'dynamic_viscosity': [['M'], ['L', 'T']],
                               'shear_rate': [[], ['T']],
                               'shear_stress': [['M'], ['L', 'T', 'T']]}
        if derived not in valid_derived_types.keys():
            raise ValueError("Invalid type: {}.".format(derived))
        else:
            self.base = derived
        self.dimensions = valid_derived_types[derived]
        numerator = 1
        denominator = 1
        map = {'L': base_unit_system.length.unit,
               'T': base_unit_system.time.unit,
               'M': base_unit_system.mass.unit}
        for dim in self.dimensions[0]:
            numerator *= map[dim]
        for dim in self.dimensions[1]:
            denominator *= map[dim]
        self.unit = numerator / denominator

    def __str__(self):
        output = "Derived Unit: {}\n".format(self.base)
        output += "------------\n"
        output += "{}\n".format(self.unit)
        output += "------------"
        return output

    def __mul__(self, other):
        if isinstance(other, DerivedUnit):
            return self.unit * other.unit

    def __truediv__(self, other):
        if isinstance(other, DerivedUnit):
            return self.unit / other.unit


class UnitSystem(object):
    def __init__(self, **kwargs):
        """
        The Units class defines the units that are used
        to generate a synthetic vascular tree.
        """
        # Base Units
        self.length = 'cm'
        self.time = 's'
        self.mass = 'g'
        # Derived Units
        self.pressure = None
        self.flow = None
        self.area = None
        self.volume = None
        self.velocity = None
        self.acceleration = None
        self.density = None
        self.kinematic_viscosity = None
        self.dynamic_viscosity = None
        self.shear_rate = None
        self.shear_stress = None


class ConvertBaseUnit(object):
    def __init__(self, base_unit_0, base_unit_1):
        """
        Convert between two base units.
        """
        self.factor = 1
        metric_length = {'nm': 1e-9,
                         'um': 1e-6,
                         'mm': 1e-3,
                         'cm': 1e-2,
                         'm': 1,
                         'km': 1e3}
        metric_weight = {'ng': 1e-12,
                         'ug': 1e-9,
                         'mg': 1e-6,
                         'g': 1e-3,
                         'kg': 1}
        imperial_length = {'in': 1,
                           'ft': 12,
                           'yd': 36,
                           'mi': 63360}
        imperial_weight = {'oz': 1,
                           'lb': 16,
                           'stone': 224,
                           'ton': 35840}
        metric_to_imperial = {'m_to_in': 39.3701,
                              'kg_to_oz': 35.274}
        imperial_to_metric = {'in_to_m': 0.0254,
                              'oz_to_kg': 0.0283495}
        time = {'s': 1,
                'min': 60,
                'hr': 3600,
                'day': 86400,
                'wk': 604800,
                'mo': 2.628e6,
                'yr': 31536000}
        if base_unit_0.base != base_unit_1.base:
            raise ValueError("Base units must be the same type. {} != {}.".format(base_unit_0.base,base_unit_1.base))
        base_unit_0_type = None
        base_unit_1_type = None
        if base_unit_0.base == 'length':
            if str(base_unit_0.unit) in metric_length.keys():
                base_unit_0_type = 'metric'
                self.factor *= metric_length[str(base_unit_0.unit)]
            elif str(base_unit_0.unit) in imperial_length.keys():
                base_unit_0_type = 'imperial'
                self.factor *= imperial_length[str(base_unit_0.unit)]
            else:
                raise ValueError("Invalid first unit: {}.".format(base_unit_0.unit))
            if str(base_unit_1.unit) in metric_length.keys():
                base_unit_1_type = 'metric'
                if base_unit_0_type == base_unit_1_type:
                    self.factor *= 1/metric_length[str(base_unit_1.unit)]
                else:
                    self.factor *= imperial_to_metric['in_to_m']
                    self.factor *= 1/imperial_length[str(base_unit_1.unit)]
            elif str(base_unit_1.unit) in imperial_length.keys():
                base_unit_1_type = 'imperial'
                if base_unit_0_type == base_unit_1_type:
                    self.factor *= 1/imperial_length[str(base_unit_1.unit)]
                else:
                    self.factor *= metric_to_imperial['m_to_in']
                    self.factor *= 1/imperial_length[str(base_unit_1.unit)]
            else:
                raise ValueError("Invalid second unit: {}.".format(base_unit_1.unit))

