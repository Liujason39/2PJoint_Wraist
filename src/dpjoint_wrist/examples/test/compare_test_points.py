"""Compare measured wrist points with geometry.json using the package API.

Run from the repository root:
    .venv/bin/python src/dpjoint_wrist/examples/test/compare_test_points.py

Slides 5-3/5-8 map to actuator indices 0/1; joints 6/7 equal q[1]/-q[0].
Table strokes are mm and joint angles are assumed to be degrees. The package
uses total actuator lengths in metres and angles in radians.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Support direct execution from a source checkout, even without installation.
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from dpjoint_wrist import Wrist


# command 5-3, command 5-8, actual 5-3, actual 5-8, joint 6, joint 7
POINTS = np.array([
    [0, 0, 0, 0, 0, 0],
    [-10.0, -10.0, -10.1222, -9.8746, -0.2109, -29.3966],
    [-5.4, -4.8, -5.3556, -4.8437, -0.4245, -14.6551],
    [3.4, -10.0, 3.2739, -10.0417, 11.1632, -9.8845],
    [1.4, -5.2, 1.4141, -5.1947, 5.5153, -5.4358],
    [-0.2, 3.0, -0.2173, 2.9822, -2.6682, 4.0021],
    [-1.8, 5.0, -1.7844, 5.0053, -5.6693, 4.6864],
    [-3.6, 10.0, -3.5554, 10.0155, -11.3869, 9.6036],
    [4.8, 5.6, 4.8375, 5.5600, -0.6010, 15.3066],
    [9.8, 10.0, 9.7029, 10.0963, -0.3320, 30.5031],
])


def print_table(title, columns, rows):
    print(f"\n{title}")
    print("  ".join(f"{name:>12}" for name in columns))
    for index, row in enumerate(rows):
        print(f"{index:12d}  " + "  ".join(f"{value:12.4f}" for value in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "geometry.json")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        wrist = Wrist(**json.load(stream))

    command, actual, measured_joints = POINTS[:, :2], POINTS[:, 2:4], POINTS[:, 4:]
    print(f"Geometry: {args.config.resolve()}")
    print("Mapping: slide 5-3 -> actuator 0; slide 5-8 -> actuator 1")
    print("         joint 6 = q[1]; joint 7 = -q[0] (degrees)")
    print("IK mapping: q = [-joint 7, joint 6], converted to radians.")
    print("FK uses from_stroke(mm / 1000); IK uses to_stroke(ik(deg2rad(q))).")
    print("Each independent FK point is seeded at the configured home pose.")
    print("Errors are prediction - measurement; tracking is actual - command.")
    print("nan denotes a failed solve, reported below; no acceptance tolerance supplied.")
    print_table("Slide tracking (mm)",
                ["id", "cmd_5-3", "cmd_5-8", "act_5-3", "act_5-8", "err_5-3", "err_5-8"],
                np.column_stack([command, actual, actual - command]))

    failures = []
    for label, strokes in [("command", command), ("actual", actual)]:
        predicted = np.full_like(measured_joints, np.nan)
        for index, stroke in enumerate(strokes):
            try:
                rho = wrist.from_stroke(stroke / 1000)
                q_deg = np.rad2deg(wrist.fk(rho, seed=wrist.home))
                predicted[index] = [q_deg[1], -q_deg[0]]
            except ValueError as exc:
                failures.append(f"FK {label}, point {index}: {exc}")
        print_table(f"FK from {label} strokes vs measured joints (deg)",
                    ["id", "meas_j6", "meas_j7", "FK_j6", "FK_j7", "err_j6", "err_j7"],
                    np.column_stack([measured_joints, predicted, predicted - measured_joints]))
        valid = np.all(np.isfinite(predicted), axis=1)
        if valid.any():
            errors = predicted[valid] - measured_joints[valid]
            print(f"Successful points: {valid.sum()}/{len(strokes)}; "
                  f"RMSE j6/j7 (deg): {np.sqrt(np.mean(errors**2, axis=0))}; "
                  f"max absolute error: {np.max(np.abs(errors), axis=0)}")

    predicted_stroke = np.full_like(actual, np.nan)
    for index, joints in enumerate(measured_joints):
        try:
            q = np.deg2rad([-joints[1], joints[0]])
            predicted_stroke[index] = wrist.to_stroke(wrist.ik(q)) * 1000
        except ValueError as exc:
            failures.append(f"IK, point {index}: {exc}")
    print_table("IK from measured joints vs actual strokes (mm)",
                ["id", "act_5-3", "act_5-8", "IK_5-3", "IK_5-8", "err_5-3", "err_5-8"],
                np.column_stack([actual, predicted_stroke, predicted_stroke - actual]))
    print(f"\nSolver failures: {len(failures)}")
    for failure in failures:
        print(f"  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
