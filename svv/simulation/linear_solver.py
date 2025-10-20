from xml.dom import minidom


class LinearSolverBase:
    def __init__(self):
        self.solver_type = "NS"
        self.absolute_tolerance = 1.0e-10
        self.tolerance = 0.5
        self.max_iterations = 1000
        # Linear algebra backend and options used by svMultiPhysics.
        # These control the required <Linear_algebra> section emitted
        # under the <LS> element.
        self.linear_algebra_type = "fsils"  # fsils | petsc | trilinos
        self.preconditioner = None          # e.g., fsils, petsc-rcs, trilinos-ilut
        self.assembly = None                # Optional: none | fsils | petsc | trilinos
        self.ns_cg_max_iterations = 1000
        self.ns_cg_tolerance = 1.0e-2
        self.ns_gm_max_iterations = 1000
        self.ns_gm_tolerance = 1.0e-2
        self.use_trilinos_for_assembly = False
        self.krylov_space_dimension = 50

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(vars(self))

    def __eq__(self, other):
        check = ["solver_type", "absolute_tolerance", "tolerance", "max_iterations", "preconditioner",
                 "ns_cg_max_iterations", "ns_cg_tolerance", "ns_gm_max_iterations", "ns_gm_tolerance",
                 "use_trilinos_for_assembly"]
        attributes = vars(self)
        other_attributes = vars(other)
        return all(other_attributes[key] == attributes[key] for key in check)

    def __ne__(self, other):
        return not self.__eq__(other)


class LinearSolver(LinearSolverBase):
    def __init__(self):
        super().__init__()
        self.file = minidom.Document()

    def __str__(self):
        return self.toxml().toprettyxml()

    def __repr__(self):
        try:
            return self.toxml().toprettyxml(indent="  ")
        except Exception:
            return str(vars(self))

    def set_type(self, solver_type):
        """
        Set the linear solver type. Options are 'CG', 'GMRES', 'NS', 'BICGS'.
        Note that not all linear solvers and preconditioners are compatible.
        """
        if solver_type not in ["CG", "GMRES", "NS", "BICGS"]:
            raise ValueError("Solver type must be 'CG' or 'GMRES'.")
        self.solver_type = solver_type

    def set_linear_algebra_type(self, linalg_type):
        """
        Set the linear algebra backend type used by svMultiPhysics.
        Valid values: 'fsils', 'petsc', 'trilinos'.
        """
        if not isinstance(linalg_type, str):
            raise ValueError("Linear algebra type must be a string.")
        if linalg_type.lower() not in ["fsils", "petsc", "trilinos"]:
            raise ValueError("Linear algebra type must be 'fsils', 'petsc', or 'trilinos'.")
        self.linear_algebra_type = linalg_type.lower()

    def set_preconditioner(self, preconditioner):
        """
        Set the linear algebra preconditioner string. This is passed
        through to svMultiPhysics and must be compatible with the
        selected backend (e.g., 'fsils', 'petsc-rcs', 'trilinos-ilut').
        """
        if not isinstance(preconditioner, str):
            raise ValueError("Preconditioner must be a string.")
        self.preconditioner = preconditioner

    def set_linear_algebra(self, linalg_type=None, preconditioner=None, assembly=None):
        """
        Convenience method to set linear algebra backend, preconditioner,
        and optional assembly type.
        """
        if isinstance(linalg_type, str):
            self.set_linear_algebra_type(linalg_type)
        if isinstance(preconditioner, str):
            self.set_preconditioner(preconditioner)
        if isinstance(assembly, str):
            self.assembly = assembly

    def toxml(self):
        ls = self.file.createElement("LS")
        if not isinstance(self.solver_type, type(None)):
            ls.setAttribute("type", self.solver_type)
        else:
            raise ValueError("Solver type must be set.")

        # Emit required <Linear_algebra> section for svMultiPhysics.
        if not isinstance(self.linear_algebra_type, type(None)):
            linear_algebra = self.file.createElement("Linear_algebra")
            linear_algebra.setAttribute("type", self.linear_algebra_type)
            if isinstance(self.preconditioner, str) and len(self.preconditioner) > 0:
                precond = self.file.createElement("Preconditioner")
                precond.appendChild(self.file.createTextNode(self.preconditioner))
                linear_algebra.appendChild(precond)
            if isinstance(self.assembly, str) and len(self.assembly) > 0:
                assembly = self.file.createElement("Assembly")
                assembly.appendChild(self.file.createTextNode(self.assembly))
                linear_algebra.appendChild(assembly)
            ls.appendChild(linear_algebra)

        if not isinstance(self.absolute_tolerance, type(None)):
            absolute_tolerance = self.file.createElement("Absolute_tolerance")
            absolute_tolerance.appendChild(self.file.createTextNode(str(self.absolute_tolerance)))
            ls.appendChild(absolute_tolerance)

        if not isinstance(self.tolerance, type(None)):
            tolerance = self.file.createElement("Tolerance")
            tolerance.appendChild(self.file.createTextNode(str(self.tolerance)))
            ls.appendChild(tolerance)

        if not isinstance(self.max_iterations, type(None)):
            max_iterations = self.file.createElement("Max_iterations")
            max_iterations.appendChild(self.file.createTextNode(str(self.max_iterations)))
            ls.appendChild(max_iterations)

        #if not isinstance(self.preconditioner, type(None)):
        #    preconditioner = self.file.createElement("Preconditioner")
        #    preconditioner.appendChild(self.file.createTextNode(self.preconditioner))
        #    ls.appendChild(preconditioner)

        if not isinstance(self.ns_cg_max_iterations, type(None)):
            ns_cg_max_iterations = self.file.createElement("NS_CG_max_iterations")
            ns_cg_max_iterations.appendChild(self.file.createTextNode(str(self.ns_cg_max_iterations)))
            ls.appendChild(ns_cg_max_iterations)

        if not isinstance(self.ns_cg_tolerance, type(None)):
            ns_cg_tolerance = self.file.createElement("NS_CG_tolerance")
            ns_cg_tolerance.appendChild(self.file.createTextNode(str(self.ns_cg_tolerance)))
            ls.appendChild(ns_cg_tolerance)

        if not isinstance(self.ns_gm_max_iterations, type(None)):
            ns_gm_max_iterations = self.file.createElement("NS_GM_max_iterations")
            ns_gm_max_iterations.appendChild(self.file.createTextNode(str(self.ns_gm_max_iterations)))
            ls.appendChild(ns_gm_max_iterations)

        if not isinstance(self.ns_gm_tolerance, type(None)):
            ns_gm_tolerance = self.file.createElement("NS_GM_tolerance")
            ns_gm_tolerance.appendChild(self.file.createTextNode(str(self.ns_gm_tolerance)))
            ls.appendChild(ns_gm_tolerance)

        if self.use_trilinos_for_assembly:
            use_trilinos_for_assembly = self.file.createElement("Use_trilinos_for_assembly")
            use_trilinos_for_assembly.appendChild(self.file.createTextNode("true"))
            ls.appendChild(use_trilinos_for_assembly)

        if not isinstance(self.krylov_space_dimension, type(None)):
            krylov_space_dimension = self.file.createElement("Krylov_space_dimension")
            krylov_space_dimension.appendChild(self.file.createTextNode(str(self.krylov_space_dimension)))
            ls.appendChild(krylov_space_dimension)
        return ls
