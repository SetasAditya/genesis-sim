"""Isolate the estimated hollow insert collider from robot and room dynamics."""

import json
import numpy as np
from wetlab_isaaclab.assets import PROJECT_ROOT as ROOT
from isaacsim import SimulationApp

app = SimulationApp(
    {"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
)
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from wetlab_isaaclab.sandbox.asset_builder import build_assets

out = ROOT / "outputs/sandbox"
report = {"passed": False, "robot_execution": False, "scope": "isolated_insert_fixture"}
try:
    assets = out / "isolated_assets"
    build_assets(assets, ROOT / "assets/htr")
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")

    def spawn(name, file, semantic):
        p = stage.DefinePrim("/World/" + name, "Xform")
        p.GetReferences().AddReference(str(file))
        box = (
            UsdGeom.BBoxCache(0, ["default", "render"])
            .ComputeUntransformedBound(p)
            .ComputeAlignedRange()
        )
        low, high = np.array(box.GetMin()), np.array(box.GetMax())
        origin = np.r_[(low[:2] + high[:2]) / 2, low[2]]
        UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(*(semantic - origin)))
        return p, origin

    well, _ = spawn(
        "well", ROOT / "assets/htr/HTR_Well.usda", np.array([0.0, 0.0, 0.02])
    )
    UsdPhysics.RigidBodyAPI(well).CreateKinematicEnabledAttr(True)
    target = np.array([-0.045, -0.026, 0.0249])
    insert, origin = spawn(
        "insert", assets / "insert_clearance.usda", target + [0, 0, 0.003]
    )
    for p in Usd.PrimRange(well):
        if p.IsA(UsdGeom.Mesh):
            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p).CreateSdfResolutionAttr(1024)
            a = PhysxSchema.PhysxCollisionAPI.Apply(p)
            a.CreateContactOffsetAttr(0.00002)
            a.CreateRestOffsetAttr(0.0)
    a = PhysxSchema.PhysxRigidBodyAPI.Apply(insert)
    a.CreateLinearDampingAttr(1.0)
    a.CreateAngularDampingAttr(1.0)
    a.CreateMaxDepenetrationVelocityAttr(0.02)
    a.CreateSolverPositionIterationCountAttr(64)
    a.CreateSolverVelocityIterationCountAttr(8)
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 240, rendering_dt=1 / 60)
    body = world.scene.add(
        SingleRigidPrim("/World/insert", name="insert", reset_xform_properties=False)
    )
    world.reset()
    samples = []
    for i in range(2400):
        world.step(render=False)
        if i >= 2280:
            samples.append(
                (
                    float(np.linalg.norm(body.get_linear_velocity())),
                    float(np.linalg.norm(body.get_angular_velocity())),
                )
            )
    p, q = body.get_world_pose()
    error = np.array(p) + origin - target
    peak = np.max(samples, axis=0)
    report.update(
        {
            "seconds": 10.0,
            "error_m": error.tolist(),
            "peak_last_half_second_linear_speed_m_s": float(peak[0]),
            "peak_last_half_second_angular_speed_rad_s": float(peak[1]),
            "estimated_collision_diameter_mm": 19.55,
            "visual_diameter_mm": 19.75,
            "well_held_as_fixture": True,
        }
    )
    report["passed"] = bool(
        np.isfinite(error).all()
        and np.linalg.norm(error[:2]) < 0.0002
        and abs(error[2]) < 0.0005
        and peak[0] < 0.01
        and peak[1] < 0.1
    )
except Exception as e:
    import traceback

    report["error"] = str(e)
    report["traceback"] = traceback.format_exc()
finally:
    (out / "insert_isolated.json").write_text(json.dumps(report, indent=2))
    print("INSERT_ISOLATED", json.dumps(report), flush=True)
    app.close()
raise SystemExit(0 if report["passed"] else 2)
