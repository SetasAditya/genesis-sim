"""Direct fixture seating diagnostics; these are explicitly not robot trials."""

import json
import sys
import numpy as np
from wetlab_isaaclab.assets import PROJECT_ROOT as ROOT

REPORT_NAME = "insert_fit.json" if "--insert-only" in sys.argv else "fixture_fits.json"
from isaacsim import SimulationApp

app = SimulationApp(
    {"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
)
from wetlab_isaaclab.sandbox.environment import WetLabSandbox
from wetlab_isaaclab.sandbox.config import Action

report = {
    "test_kind": "direct_fixture_seating",
    "robot_execution": False,
    "checks": [],
    "passed": False,
}
try:
    env = WetLabSandbox(ROOT, "petri", render=False)
    report["insert_collision"] = env.insert_collision

    def settle(seconds=3.0):
        for _ in range(round(seconds * 30)):
            env.step(Action())

    def seat(name, target, xy_tolerance=0.003, z_tolerance=0.003):
        # Diagnostic placement only; release under gravity with no attachment.
        env.debug_set_pose(name, np.asarray(target) + [0, 0, 0.003])
        settle()
        s = env.object_states()[name]
        for _ in range(2):
            if s.linear_speed < 0.01 and s.angular_speed < 0.1:
                break
            settle(2.0)
            s = env.object_states()[name]
        d = np.asarray(s.position) - target
        passed = bool(
            np.isfinite(d).all()
            and np.linalg.norm(d[:2]) < xy_tolerance
            and abs(d[2]) < z_tolerance
            and s.linear_speed < 0.01
            and s.angular_speed < 0.1
            and s.up_dot > 0.995
        )
        result = {
            "object": name,
            "target_m": list(target),
            "actual_m": list(s.position),
            "error_m": d.tolist(),
            "linear_speed_m_s": s.linear_speed,
            "angular_speed_rad_s": s.angular_speed,
            "up_dot": s.up_dot,
            "xy_tolerance_m": xy_tolerance,
            "z_tolerance_m": z_tolerance,
            "passed": passed,
        }
        report["checks"].append(result)
        print("FIXTURE", json.dumps(result), flush=True)
        (env.output / REPORT_NAME).write_text(json.dumps(report, indent=2))

    if "--insert-only" in sys.argv:
        report["scope"] = "insert_fit"
        env.reset(0, "reactor_assembly")
        for name in ("basket", "well", "insert"):
            seat(name, env.targets[name], 0.002, 0.0005 if name == "insert" else 0.003)
        report["passed"] = all(c["passed"] for c in report["checks"])
        raise SystemExit(0 if report["passed"] else 2)
    env.reset(0, "petri", mode="close")
    seat("lid", env.targets["lid_seated"], 0.0006, 0.002)
    env.reset(0, "tube_insertion")
    report["rack_openings"] = env.check_openings()
    seat("tube", env.targets["tube_seated"], 0.0003, 0.002)
    env.reset(0, "fridge_retrieval")
    # Isolate the passive door mechanism; park the robot clear of its swept arc.
    env.robot.set_world_pose(position=np.array([-5.8, 1.2, 0.075]))
    env.controller.base_anchor = None
    from scipy.spatial.transform import Rotation

    for i in range(960):
        _, q = env._pose("fridge_door")
        angle = abs(
            float(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2])
        )
        if angle > np.deg2rad(75):
            break
        env.handles["fridge_door"]._rigid_prim_view.apply_forces_and_torques_at_pos(
            torques=np.array([[0.0, 0.0, -1.0]]), is_global=True
        )
        env.world.step(render=False)
    settle(1.0)
    _, q = env._pose("fridge_door")
    angle = abs(float(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2]))
    report["checks"].append(
        {
            "mechanism": "passive_fridge_hinge",
            "angle_rad": angle,
            "direct_diagnostic_torque_Nm": -1.0,
            "passed": bool(np.deg2rad(65) < angle < np.deg2rad(111)),
        }
    )
    env.reset(0, "pipetting")
    env.pipette_art.set_joint_positions(np.array([-0.006]))
    from isaacsim.core.utils.types import ArticulationAction

    env.pipette_art.apply_action(ArticulationAction(joint_positions=np.array([0.0])))
    env.pipette_art.set_joint_velocities(np.array([0.0]))
    settle(2.0)
    displacement = float(env.pipette_art.get_joint_positions()[0])
    report["checks"].append(
        {
            "mechanism": "spring_return_plunger",
            "initial_displacement_m": -0.006,
            "released_displacement_m": displacement,
            "passed": bool(abs(displacement) < 0.0004),
        }
    )
    env.reset(0, "reactor_assembly")
    for name in (
        "basket",
        "well",
        "insert",
        "cover",
        "bolt0",
        "bolt1",
        "bolt2",
        "bolt3",
    ):
        seat(
            name,
            env.targets[name],
            0.0006 if name.startswith("bolt") else 0.002,
            0.0005
            if name == "insert"
            else 0.00075
            if name.startswith("bolt")
            else 0.003,
        )
    report["passed"] = all(c["passed"] for c in report["checks"]) and all(
        c["passed"] for c in report["rack_openings"]
    )
except Exception as e:
    import traceback

    report["error"] = str(e)
    report["traceback"] = traceback.format_exc()
finally:
    path = ROOT / "outputs/sandbox" / REPORT_NAME
    path.write_text(json.dumps(report, indent=2))
    print("FIXTURE_REPORT", json.dumps(report), flush=True)
    app.close()
raise SystemExit(0 if report["passed"] else 2)
