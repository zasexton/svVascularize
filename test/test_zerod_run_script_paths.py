import json
import os
import shlex
import stat
import subprocess
import sys

from svv.simulation.fluid.rom.zero_d.process import DEFAULT_OUTPUT_FILENAME, run_0d_script


def _write_fake_solver(tmp_path):
    impl = tmp_path / "fake_solver_impl.py"
    impl.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import sys",
                "",
                "payload = {'argv': sys.argv[1:], 'cwd': os.getcwd()}",
                "with open(os.path.join(os.getcwd(), 'solver_invocation.json'), 'w', encoding='utf-8') as handle:",
                "    json.dump(payload, handle)",
                "with open(sys.argv[2], 'w', encoding='utf-8') as handle:",
                "    handle.write('name,time,flow_in,flow_out,pressure_in,pressure_out\\n')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if os.name == "nt":
        solver = tmp_path / "fake_solver.bat"
        solver.write_text(f'@echo off\n"{sys.executable}" "{impl}" %*\n', encoding="utf-8")
        return solver

    solver = tmp_path / "fake_solver"
    solver.write_text(
        "#!/usr/bin/env sh\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(impl))} \"$@\"\n",
        encoding="utf-8",
    )
    solver.chmod(solver.stat().st_mode | stat.S_IXUSR)
    return solver


def test_generated_run_script_writes_output_next_to_script_when_launched_elsewhere(tmp_path):
    export_dir = tmp_path / "case"
    launch_dir = tmp_path / "elsewhere"
    export_dir.mkdir()
    launch_dir.mkdir()

    solver = _write_fake_solver(tmp_path)
    input_path = export_dir / "solver_0d.in"
    output_path = export_dir / DEFAULT_OUTPUT_FILENAME
    input_path.write_text("{}", encoding="utf-8")
    (export_dir / "run.py").write_text(
        run_0d_script.format(
            solver_exe=str(solver),
            input_filename=input_path.name,
            output_filename=DEFAULT_OUTPUT_FILENAME,
        ),
        encoding="utf-8",
    )

    subprocess.check_call([sys.executable, str(export_dir / "run.py")], cwd=launch_dir)

    assert output_path.is_file()
    assert not (launch_dir / DEFAULT_OUTPUT_FILENAME).exists()

    payload = json.loads((export_dir / "solver_invocation.json").read_text(encoding="utf-8"))
    assert payload["cwd"] == str(export_dir)
    assert payload["argv"] == [str(input_path), str(output_path)]
