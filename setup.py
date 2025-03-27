from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import sys
import os
import subprocess
from urllib.request import urlretrieve
import shutil
import tarfile
import multiprocessing

num_cores = multiprocessing.cpu_count() // 2

from setuptools.command.build_ext import build_ext

class DownloadAndBuildExt(build_ext):
    def run(self):
        """
        # -----------------------------
        # 1. Download: handle errors
        # -----------------------------
        download_url_0d = "https://github.com/SimVascular/svZeroDSolver/archive/refs/tags/v2.0.tar.gz"
        download_url_1d = "https://github.com/SimVascular/svOneDSolver/archive/refs/tags/c9caded.tar.gz"
        download_url_3d = "https://github.com/SimVascular/svMultiPhysics/archive/refs/tags/March_2025.tar.gz"

        tarball_path_0d = "svZeroDSolver.tar.gz"
        tarball_path_1d = "svOneDSolver.tar.gz"
        tarball_path_3d = "svMultiPhysics.tar.gz"

        try:
            if not os.path.exists(tarball_path_0d):
                print(f"Downloading {download_url_0d}...")
                urlretrieve(download_url_0d, tarball_path_0d)
            if not os.path.exists(tarball_path_1d):
                print(f"Downloading {download_url_1d}...")
                urlretrieve(download_url_1d, tarball_path_1d)
            if not os.path.exists(tarball_path_3d):
                print(f"Downloading {download_url_3d}...")
                urlretrieve(download_url_3d, tarball_path_3d)
        except Exception as e:
            raise RuntimeError("Error downloading solver archives.") from e

        # -----------------------------
        # 2. Extract: handle errors
        # -----------------------------
        source_path_0d = os.path.abspath("svZeroDSolver")
        source_path_1d = os.path.abspath("svOneDSolver")
        source_path_3d = os.path.abspath("svMultiPhysics")

        try:
            if not os.path.exists(source_path_0d):
                with tarfile.open(tarball_path_0d, "r:gz") as t:
                    t.extractall(source_path_0d)
            if not os.path.exists(source_path_1d):
                with tarfile.open(tarball_path_1d, "r:gz") as t:
                    t.extractall(source_path_1d)
            if not os.path.exists(source_path_3d):
                with tarfile.open(tarball_path_3d, "r:gz") as z:
                    z.extractall(source_path_3d)
        except Exception as e:
            raise RuntimeError("Error extracting solver archives.") from e

        # -----------------------------
        # 3. Configure & build with CMake
        # -----------------------------
        build_dir    = os.path.abspath("bin")
        build_dir_0d = os.path.abspath("bin/solver-0d")
        build_dir_1d = os.path.abspath("bin/solver-1d")
        build_dir_3d = os.path.abspath("bin/solver-3d")

        os.makedirs(build_dir_0d, exist_ok=True)
        os.makedirs(build_dir_1d, exist_ok=True)
        os.makedirs(build_dir_3d, exist_ok=True)

        # For each solver, configure and build. Catch errors if subprocess fails.
        try:
            subprocess.check_call(["cmake", f"-B{build_dir_0d}", f"-S{os.path.join(source_path_0d, 'svZeroDSolver-2.0')}"])
            subprocess.check_call(["cmake", f"-B{build_dir_1d}", f"-S{os.path.join(source_path_1d, 'svOneDSolver-c9caded')}"])
            subprocess.check_call(["cmake", f"-B{build_dir_3d}", f"-S{os.path.join(source_path_3d, 'svMultiPhysics-March_2025')}"])

            subprocess.check_call(["cmake", "--build", build_dir_0d, "--parallel", str(num_cores)])
            subprocess.check_call(["cmake", "--build", build_dir_1d, "--parallel", str(num_cores)])
            subprocess.check_call(["cmake", "--build", build_dir_3d, "--parallel", str(num_cores)])
        except subprocess.CalledProcessError as e:
            raise RuntimeError("CMake configure or build failed for one of the solvers.") from e

        # -----------------------------
        # 4. Verify executables
        # -----------------------------
        solver_0d_name = "svzerodsolver"
        solver_1d_name = "OneDSolver"
        solver_3d_name = "svmultiphysics"

        solver_0d_path = os.path.join(build_dir_0d, solver_0d_name)
        solver_1d_path = os.path.join(build_dir_1d, "bin", solver_1d_name)
        solver_3d_path = os.path.join(build_dir_3d, "svMultiPhysics-build", "bin", solver_3d_name)
        # If the build failed or never produced the exe, raise an error
        missing = []
        if not os.path.isfile(solver_0d_path):
            missing.append(solver_0d_path)
        if not os.path.isfile(solver_1d_path):
            missing.append(solver_1d_path)
        if not os.path.isfile(solver_3d_path):
            missing.append(solver_3d_path)

        if missing:
            raise RuntimeError(
                "ERROR: Some solvers did not build correctly. Missing:\n  "
                + "\n  ".join(missing)
            )

        # --------------------------------
        # 5. Copy executables into package
        # --------------------------------
        solvers_dir = os.path.join("svv", "solvers")
        os.makedirs(solvers_dir, exist_ok=True)
        try:
            shutil.copy2(solver_0d_path, os.path.join(solvers_dir, solver_0d_name))
            shutil.copy2(solver_1d_path, os.path.join(solvers_dir, solver_1d_name))
            shutil.copy2(solver_3d_path, os.path.join(solvers_dir, solver_3d_name))
        except Exception as e:
            raise RuntimeError("Failed copying solver executables to package directory.") from e

        # ----------------------------------
        # 6. CLEANUP if everything succeeded
        # ----------------------------------
        # We'll remove archives + source folders now that we have the built executables.
        try:
            # Remove the archives
            if os.path.isfile(tarball_path_0d):
                os.remove(tarball_path_0d)
            if os.path.isfile(tarball_path_1d):
                os.remove(tarball_path_1d)
            if os.path.isfile(tarball_path_3d):
                os.remove(tarball_path_3d)

            # Remove the source directories
            if os.path.isdir(source_path_0d):
                shutil.rmtree(source_path_0d, ignore_errors=True)
            if os.path.isdir(source_path_1d):
                shutil.rmtree(source_path_1d, ignore_errors=True)
            if os.path.isdir(source_path_3d):
                shutil.rmtree(source_path_3d, ignore_errors=True)

            # Remove the temporary build directories
            if os.path.isdir(build_dir):
                shutil.rmtree(build_dir, ignore_errors=True)

            print("Cleanup complete: Removed source folders, downloaded archives, and temporary build directories.")
        except Exception as e:
            # Usually you might just print a warning. We'll raise to be explicit.
            raise RuntimeError("Warning: Cleanup of archives/folders failed.") from e

        # -----------------------------------
        # 7. Let the normal build_ext proceed
        # -----------------------------------
        """
        super().run()

    def finalize_options(self):
        super().finalize_options()
        self.inplace = True

def get_extra_compile_args():
    extra_args = []
    if sys.platform == "win32":
        # For MSVC or clang-cl on Windows you have to choose one of the /arch options.
        cpuinfo = None
        if cpuinfo is not None:
            info = cpuinfo.get_cpu_info()
            flags = info.get("flags", [])
            # Check for AVX512 first.
            if "avx512f" in flags:
                # This flag is supported on clang-cl and may be available in your MSVC environment.
                extra_args.extend(["/O2", "/arch:AVX512"])
            elif "avx2" in flags:
                extra_args.extend(["/O2", "/arch:AVX2"])
            elif "avx" in flags:
                extra_args.extend(["/O2", "/arch:AVX"])
            else:
                # SSE2 is the minimum on most modern CPUs.
                extra_args.extend(["/O2", "/arch:SSE2"])
            update_compile_args = ['/openmp']
            update_link_args = []
        else:
            # If cpuinfo isn't available, fall back to a safe default.
            extra_args.extend(["/O2", "/arch:SSE2"])
            update_compile_args = ['/openmp']
            update_link_args = []
    else:
        # For GCC/Clang on Linux/macOS, -march=native will enable all available features.
        extra_args.extend(["-O3", "-march=native", "-ftree-vectorizer-verbose=2"])
        update_compile_args = ['-fopenmp']
        update_link_args = ['-fopenmp']
    return extra_args, update_compile_args, update_link_args

#extra_compile_args, update_compile_args, update_link_args = get_extra_compile_args()
# extra_compile_args=update_compile_args, extra_link_args=update_link_args),
extensions = [
    Extension('svv.domain.routines.c_allocate', ['svv/domain/routines/c_allocate.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.domain.routines.c_sample', ['svv/domain/routines/c_sample.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.utils.spatial.c_distance', ['svv/utils/spatial/c_distance.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.tree.utils.c_angle', ['svv/tree/utils/c_angle.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.tree.utils.c_basis', ['svv/tree/utils/c_basis.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.tree.utils.c_close', ['svv/tree/utils/c_close.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.tree.utils.c_local_optimize', ['svv/tree/utils/c_local_optimize.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.tree.utils.c_obb', ['svv/tree/utils/c_obb.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.tree.utils.c_update', ['svv/tree/utils/c_update.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.tree.utils.c_extend', ['svv/tree/utils/c_extend.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.simulation.utils.close_segments', ['svv/simulation/utils/close_segments.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
    Extension('svv.simulation.utils.extract', ['svv/simulation/utils/extract.pyx'],
              include_dirs=[numpy.get_include()], language='c++'),
]

__version__ = '0.0.29'

with open("README.md", "r", encoding="utf-8") as file:
    DESCRIPTION = file.read()

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3.9',
               'Programming Language :: Python :: 3.10',
               'Programming Language :: Python :: 3.11',
               'Programming Language :: Python :: 3.12',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX :: Linux',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS']

PACKAGES = find_packages(include=['svv', 'svv.*'])

REQUIREMENTS = ["numpy<=1.24.0",
                "scipy>=1.10.1",
                "matplotlib>=3.7.5",
                "Cython>=3.0.7",
                "usearch>=2.0.0",
                "scikit-image",
                "tetgen",
                "trimesh[all]",
                "hnswlib",
                "pyvista==0.44.2",
                "scikit-learn",
                "tqdm",
                "pymeshfix==0.17.0",
                "numexpr"]

setup_info = dict(
    name='svv',
    version=__version__,
    author='Zachary Sexton',
    author_email='zsexton@stanford.edu',
    license='MIT',
    python_requires='>=3.9',
    classifiers=CLASSIFIERS,
    packages=PACKAGES,
    description="Synthetic vascular generation, modeling, and simulation package",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    zip_safe=False,
    install_requires=REQUIREMENTS,
    cmdclass={
        "build_ext": DownloadAndBuildExt
    },
)

setup(**setup_info)
