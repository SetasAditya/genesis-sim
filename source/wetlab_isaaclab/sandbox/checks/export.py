"""Reopen the composed USD and check that every saved rigid body remains valid."""

import json
import numpy as np
from wetlab_isaaclab.assets import PROJECT_ROOT as ROOT
from isaacsim import SimulationApp

app = SimulationApp(
    {"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
)
import omni.usd
from pxr import Usd, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from wetlab_isaaclab.sandbox.robot import GalbotController
from wetlab_isaaclab.sandbox.config import Action

out = ROOT / "outputs/sandbox"
report = {"passed": False, "saved_scene_reopened": False}
try:
    omni.usd.get_context().open_stage(str(out / "all_tasks.usda"))
    stage = omni.usd.get_context().get_stage()
    report["saved_scene_reopened"] = True
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 240, rendering_dt=1 / 60)
    robot = world.scene.add(
        SingleArticulation(
            "/World/Galbot/OmniChassis/base_link",
            name="galbot",
            reset_xform_properties=False,
        )
    )
    world.reset()
    controller = GalbotController(stage, robot)
    initial = robot.get_world_pose()[0].copy()
    for i in range(2400):
        if i % 8 == 0:
            controller.command(Action(), 1 / 30)
        world.step(render=False)
    final = robot.get_world_pose()[0]
    bodies = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Equipment")):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            p = SingleRigidPrim(
                str(prim.GetPath()),
                name="check_" + str(len(bodies)),
                reset_xform_properties=False,
            )
            p.initialize()
            position, orientation = p.get_world_pose()
            bodies[str(prim.GetPath())] = {
                "position": position.tolist(),
                "finite": bool(
                    np.isfinite(position).all() and np.isfinite(orientation).all()
                ),
            }
    report.update(
        {
            "seconds": 10.0,
            "base_horizontal_drift_m": float(np.linalg.norm((final - initial)[:2])),
            "bodies": bodies,
            "body_count": len(bodies),
        }
    )
    report["passed"] = bool(
        np.isfinite(final).all()
        and report["base_horizontal_drift_m"] < 0.05
        and len(bodies) >= 20
        and all(b["finite"] for b in bodies.values())
    )
except Exception as e:
    import traceback

    report["error"] = str(e)
    report["traceback"] = traceback.format_exc()
    traceback.print_exc()
finally:
    (out / "export_validation.json").write_text(json.dumps(report, indent=2))
    print("EXPORT_VALIDATION", json.dumps(report), flush=True)
    app.close()
raise SystemExit(0 if report["passed"] else 2)
