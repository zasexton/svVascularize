from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import sys
import os
import platform
import subprocess
from urllib.request import urlretrieve
import shutil
import tarfile
import json
import stat
import multiprocessing

num_cores = multiprocessing.cpu_count() // 2

from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

class BDistWheelCmd(_bdist_wheel):
    def run(self):
        # Ensure build_ext (which calls build_mmg) runs first
        #build_mmg()
        self.run_command("build_ext")
        super().run()
        
def get_filename_without_ext(abs_path):
    base_name = os.path.basename(abs_path)      # e.g. "file.txt"
    file_name_no_ext = os.path.splitext(base_name)[0]  # e.g. "file"
    return file_name_no_ext

def remove_directory_tree(directory_path):
    """
    Removes the directory at `directory_path` along with all its files/subdirectories.
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
    else:
        print(f"Directory does not exist or is not a directory: {directory_path}")

def find_executables(top_level_folder):
    """
    Return a list of absolute paths for all executables found
    within 'top_level_folder' (recursively).

    On Windows:
        - Treat files with certain extensions (.exe, .bat, .cmd, .ps1)
          as executables.

    On Linux/macOS:
        - Check if the file has the executable bit set.

    Parameters:
        top_level_folder (str): Path to the directory where we search for executables.

    Returns:
        list of str: A list of absolute file paths that are considered executable.
    """
    # Normalize the top folder path
    top_level_folder = os.path.abspath(top_level_folder)

    # Define criteria:
    windows_exts = {".exe", ".bat", ".cmd", ".ps1"}
    is_windows = platform.system().lower().startswith("win")

    executables = []

    for root, dirs, files in os.walk(top_level_folder):
        for filename in files:
            full_path = os.path.join(root, filename)
            if is_windows:
                # On Windows, check extension
                _, ext = os.path.splitext(filename)
                if ext.lower() in windows_exts:
                    executables.append(os.path.abspath(full_path))
            else:
                # On Linux/macOS, check executable bit
                # (Also ensure it's not a directory, just in case)
                mode = os.stat(full_path).st_mode
                if (mode & stat.S_IXUSR) and not os.path.isdir(full_path):
                    executables.append(os.path.abspath(full_path))

    return executables

def find_vs_installations():
    # Modify this path if vswhere.exe is in a different place
    vswhere_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

    if not os.path.exists(vswhere_path):
        print("vswhere.exe not found at:", vswhere_path)
        return []

    # Call vswhere to get info about all VS installations in JSON format
    vswhere_cmd = [
        vswhere_path,
        "-all",  # list all VS instances
        "-requires", "Microsoft.Component.MSBuild",  # only show VS installs with MSBuild
        "-format", "json"  # output in JSON
    ]

    result = subprocess.run(vswhere_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("vswhere failed or returned an error.")
        return []

    output = result.stdout.strip()
    if not output:
        print("No Visual Studio installations found by vswhere.")
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        print("Failed to parse vswhere JSON output.")
        return []

    return data


def pick_visual_studio_generator():
    """
    Looks up installed Visual Studio versions via vswhere and picks
    the highest known generator that CMake can use, e.g.
    - "Visual Studio 17 2022" (for VS 2022)
    - "Visual Studio 16 2019" (for VS 2019)

    Returns the generator string or None if none found.
    """
    # This dictionary maps partial version info to the CMake generator name
    # Adjust as needed if you want more logic:
    vs_generators = {
        "17.": "Visual Studio 17 2022",
        "16.": "Visual Studio 16 2019",
        "15.": "Visual Studio 15 2017",
    }

    installations = find_vs_installations()
    if not installations:
        return None

    # We'll keep track of the best (highest) version found
    best_gen = None
    best_version_num = 0.0

    for inst in installations:
        # vs["catalog"]["productDisplayVersion"] might look like "17.5.33424.131"
        version_str = inst.get("catalog", {}).get("productDisplayVersion", "")
        # Attempt to parse out the major version (17, 16, 15, etc.)
        if version_str:
            # e.g., "17.4.1" => major = 17
            try:
                major_str = version_str.split(".")[0]  # '17'
                major = int(major_str)
            except ValueError:
                major = 0  # fallback

            # If we have a known generator string for this major version, use it
            # Or you might parse vs_generators keys more systematically
            if major >= 17 and "17." in version_str:
                if major > best_version_num:
                    best_gen = vs_generators["17."]
                    best_version_num = major
            elif major == 16 and "16." in version_str:
                if major > best_version_num:
                    best_gen = vs_generators["16."]
                    best_version_num = major
            elif major == 15 and "15." in version_str:
                if major > best_version_num:
                    best_gen = vs_generators["15."]
                    best_version_num = major

    return best_gen


def build_mmg(num_cores=None):
    if num_cores is None:
        num_cores = os.cpu_count() or 1

    # Make sure cmake is on PATH
    if shutil.which("cmake") is None:
        raise RuntimeError("CMake is not installed or not on the PATH.")

    download_url_mmg = "https://github.com/MmgTools/mmg/archive/refs/tags/v5.8.0.tar.gz"
    tarball_path_mmg = "mmg.tar.gz"
    source_extract_root = "mmg"

    # Download mmg if not present
    if not os.path.exists(tarball_path_mmg):
        print(f"Downloading {download_url_mmg}...")
        urlretrieve(download_url_mmg, tarball_path_mmg)

    # Extract mmg if not already extracted
    if not os.path.exists(source_extract_root):
        print("Extracting mmg...")
        with tarfile.open(tarball_path_mmg, "r:gz") as t:
            t.extractall(source_extract_root)

    # Typically the archive extracts into a folder named "mmg-5.8.0" under "mmg/"
    subdirs = os.listdir(source_extract_root)
    if not subdirs:
        raise RuntimeError("No files found after extracting mmg archive.")
    mmg_subdir = os.path.join(source_extract_root, subdirs[0])

    # Prepare build directory
    build_dir_mmg = os.path.abspath(os.path.join("bin", "mmg"))
    os.makedirs(build_dir_mmg, exist_ok=True)

    # Build up our cmake configure command
    cmake_cmd = ["cmake"]

    # On Windows, pick a Visual Studio generator if possible
    if platform.system().lower().startswith("win"):
        vs_generator = pick_visual_studio_generator()
        if vs_generator:
            cmake_cmd += ["-G", vs_generator]
        else:
            print("No suitable Visual Studio found, falling back to default generator or NMake.")
            # cmake_cmd += ["-G", "NMake Makefiles"]  # optional fallback

    # Add standard arguments
    cmake_cmd += [
        "-DCMAKE_BUILD_TYPE=Release",
        "-B", build_dir_mmg,
        "-S", mmg_subdir
    ]

    # Run configure step
    print("Configuring mmg with CMake:", " ".join(cmake_cmd))
    try:
        subprocess.check_call(cmake_cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("CMake configure failed for mmg.") from e

    # Run build step
    build_cmd = [
        "cmake",
        "--build", build_dir_mmg,
        "--parallel", str(num_cores)
    ]

    # For multi-config generators (Visual Studio, Xcode), we must specify --config Release
    # to ensure a Release build.
    if platform.system().lower().startswith("win"):
        # or detect if vs_generator is set if you want to be more precise
        build_cmd += ["--config", "Release"]

    print("Building mmg with CMake:", " ".join(build_cmd))

    try:
        subprocess.check_call(build_cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("CMake build failed for mmg.") from e

    print("MMG build completed successfully!")

    install_tmp_prefix = os.path.join("svv", "tmp")
    os.makedirs(install_tmp_prefix, exist_ok=True)
    install_cmd = [
        "cmake",
        "--install", build_dir_mmg,
        "--prefix", os.path.abspath(install_tmp_prefix),
    ]

    # For multi-configuration generators on Windows (like Visual Studio),
    # specify `--config Release` explicitly if you built in Release mode.
    if platform.system().lower().startswith("win"):
        install_cmd += ["--config", "Release"]

    print("Installing mmg with CMake:", " ".join(install_cmd))
    subprocess.check_call(install_cmd)
    print(f"mmg executables have been installed into: {install_tmp_prefix}")

    print("Copying executables and cleaning up")
    install_prefix = os.path.join("svv", "bin")
    os.makedirs(install_prefix, exist_ok=True)
    init_file = os.path.join('svv', 'bin', '__init__.py')
    if not os.path.isfile(init_file):
        with open(init_file, 'w'):
            pass
    executables = find_executables(install_tmp_prefix)
    basenames = ["mmg2d_O3", "mmg3d_O3", "mmgs_O3"]
    executables = [exe for exe in executables if get_filename_without_ext(exe) in basenames]
    for exe in executables:
        shutil.copy2(exe, install_prefix)
    remove_directory_tree(install_tmp_prefix)

    print('Remove Source, Archive, and Build directories')
    if os.path.isfile(tarball_path_mmg):
        os.remove(tarball_path_mmg)
    remove_directory_tree(build_dir_mmg)
    remove_directory_tree(source_extract_root)


def build_0d(num_cores=None):
    if num_cores is None:
        num_cores = os.cpu_count() or 1

    # Make sure cmake is on PATH
    if shutil.which("cmake") is None:
        raise RuntimeError("CMake is not installed or not on the PATH.")

    download_url_0d = "https://github.com/SimVascular/svZeroDSolver/archive/refs/tags/v2.0.tar.gz"
    tarball_path_0d = "svZeroDSolver.tar.gz"
    source_path_0d = os.path.abspath("svZeroDSolver")

    # Build up our cmake configure command
    cmake_cmd = ["cmake"]

    try:
        if not os.path.exists(tarball_path_0d):
            print(f"Downloading {download_url_0d}...")
            urlretrieve(download_url_0d, tarball_path_0d)
    except Exception as e:
        raise RuntimeError("Error downloading solver archives.") from e

    try:
        if not os.path.exists(source_path_0d):
            with tarfile.open(tarball_path_0d, "r:gz") as t:
                t.extractall(source_path_0d)
    except Exception as e:
        raise RuntimeError("Error extracting solver archives.") from e

    build_dir_0d = os.path.abspath("tmp/solver-0d")

    # Add standard arguments
    cmake_cmd += [
        "-DCMAKE_BUILD_TYPE=Release",
        "-B", build_dir_0d,
        "-S", mmg_subdir
    ]

    # Run build step
    build_cmd = [
        "cmake",
        "--build", build_dir_0d,
        "--parallel", str(num_cores)
    ]
class DownloadAndBuildExt(build_ext):
    def run(self):
        #------------------------------
        # Get MMG Remeshing Source Code
        #------------------------------
        build_mmg()
        # -----------------------------
        # 1. Download: handle errors
        # -----------------------------
        if not platform.system().lower().startswith("win"):
            """
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
            """
            super().run()
        else:
            # -----------------------------------
            # 7. Let the normal build_ext proceed
            # -----------------------------------
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

__version__ = '0.0.32'

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

KEYWORDS = ["modeling",
            "simulation",
            "tissue-engineering",
            "3d-printing",
            "fluid-dynamics"]

setup_info = dict(
    name='svv',
    version=__version__,
    author='Zachary Sexton',
    author_email='zsexton@stanford.edu',
    license='MIT',
    python_requires='>=3.9',
    classifiers=CLASSIFIERS,
    packages=PACKAGES,
    keywords=KEYWORDS,
    description="svVascularize (svv): A synthetic vascular generation, modeling, and simulation package",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    ext_modules=cythonize(extensions),
    package_data={'svv.bin': ['*']},
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    zip_safe=False,
    install_requires=REQUIREMENTS,
    cmdclass={
        'build_ext': DownloadAndBuildExt,
        'bdist_wheel': BDistWheelCmd,
    },
)

setup(**setup_info)
