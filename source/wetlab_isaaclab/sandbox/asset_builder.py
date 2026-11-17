"""Deterministic, hollow equipment models with explicit collision primitives."""

import json
import math
import os
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
from .config import PARAMETERS

WHITE = (0.83, 0.86, 0.88)
BLUE = (0.18, 0.40, 0.60)
STEEL = (0.53, 0.56, 0.60)


class Builder:
    def __init__(self, filename, name, mass=None):
        self.stage = Usd.Stage.CreateNew(str(filename))
        self.name = name
        self.root = UsdGeom.Xform.Define(self.stage, "/" + name).GetPrim()
        self.stage.SetDefaultPrim(self.root)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        self.i = 0
        if mass is not None:
            self.rigid(self.root, mass)

    def rigid(self, p, mass):
        UsdPhysics.RigidBodyAPI.Apply(p).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
        api.CreateSolverPositionIterationCountAttr(32)
        api.CreateSolverVelocityIterationCountAttr(4)
        api.CreateMaxDepenetrationVelocityAttr(0.2)

    def shape(
        self, kind, pos, size, color=WHITE, parent=None, collision=True, rotate=None
    ):
        self.i += 1
        path = str((parent or self.root).GetPath()) + f"/shape_{self.i}"
        if kind == "box":
            obj = UsdGeom.Cube.Define(self.stage, path)
            obj.CreateSizeAttr(1.0)
            x = UsdGeom.Xformable(obj)
            x.AddTranslateOp().Set(Gf.Vec3d(*pos))
            if rotate is not None:
                x.AddRotateZOp().Set(rotate)
            x.AddScaleOp().Set(Gf.Vec3d(*size))
        else:
            obj = UsdGeom.Cylinder.Define(self.stage, path)
            obj.CreateRadiusAttr(size[0])
            obj.CreateHeightAttr(size[1])
            obj.CreateAxisAttr("Z")
            UsdGeom.Xformable(obj).AddTranslateOp().Set(Gf.Vec3d(*pos))
        obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collision:
            UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
            a = PhysxSchema.PhysxCollisionAPI.Apply(obj.GetPrim())
            a.CreateContactOffsetAttr(0.0001)
            a.CreateRestOffsetAttr(0.0)
        return obj.GetPrim()

    def ring(
        self,
        radius,
        wall,
        height,
        z=0.0,
        parent=None,
        color=WHITE,
        segments=24,
        xy=(0.0, 0.0),
    ):
        # Individually convex wall panels leave the cavity physically empty.
        r = radius - wall / 2.0
        length = 2 * r * math.tan(math.pi / segments) + wall * 0.15
        for i in range(segments):
            t = 2 * math.pi * i / segments
            self.shape(
                "box",
                (xy[0] + r * math.cos(t), xy[1] + r * math.sin(t), z + height / 2),
                (wall, length, height),
                color,
                parent,
                rotate=math.degrees(t),
            )

    def cup(self, radius, height, wall=0.001, bottom=0.001, parent=None, color=WHITE):
        self.shape("cylinder", (0, 0, bottom / 2), (radius, bottom), color, parent)
        self.ring(radius, wall, height - bottom, bottom, parent, color)

    def save(self):
        self.stage.GetRootLayer().Save()


def build_assets(destination, htr_directory=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "units": "m_kg_s",
        "source": "project-authored parametric models",
        "license": "project-local; no third-party meshes",
        "physical_properties": "representative estimates",
        "parameters": PARAMETERS,
        "assets": {},
    }

    def new(name, mass=None):
        return Builder(destination / f"{name}.usda", name, mass)

    # Preserve the native CAD visual, with a separate hollow collision proxy.
    # STEP circles: outer 19.75 mm, inner 12.25 mm, height 23.10 mm,
    # cavity depth 19.75 mm. Only the outer collision diameter is estimated.
    b = new("insert_clearance", 0.01043)
    visual = b.stage.DefinePrim("/insert_clearance/visual")
    visual_path = (
        Path(htr_directory or destination.parent / "htr").resolve() / "HTR_Insert.usda"
    )
    visual.GetReferences().AddReference(
        os.path.relpath(visual_path, destination.resolve()), "/HTR_Insert/mesh"
    )
    UsdPhysics.CollisionAPI.Apply(visual).CreateCollisionEnabledAttr(False)
    floor = b.shape("cylinder", (0, 0, -0.021425), (0.009775, 0.00335), WHITE)
    UsdGeom.Imageable(floor).CreatePurposeAttr("guide")
    for i in range(32):
        points = []
        for z in (-0.01975, 0.0):
            for r, t in [
                (0.006125, i),
                (0.009775, i),
                (0.009775, i + 1),
                (0.006125, i + 1),
            ]:
                angle = 2 * math.pi * t / 32
                points.append(Gf.Vec3f(r * math.cos(angle), r * math.sin(angle), z))
        mesh = UsdGeom.Mesh.Define(b.stage, f"/insert_clearance/wall_{i}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(
            [3, 2, 1, 0, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
        )
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
            "convexHull"
        )
        api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        api.CreateContactOffsetAttr(0.00002)
        api.CreateRestOffsetAttr(0.0)
    b.save()
    b = new("dish", 0.018)
    b.cup(0.045, 0.016)
    b.save()
    b = new("lid", 0.014)
    b.ring(0.0464, 0.0008, 0.011)
    b.shape("cylinder", (0, 0, 0.0115), (0.0464, 0.001), WHITE)
    # A detachable handling clip presents a pinch target without filling the lid.
    b.shape("box", (0, -0.048, 0.0135), (0.014, 0.010, 0.003), BLUE)
    b.shape("box", (0, -0.050, 0.025), (0.014, 0.006, 0.026), BLUE)
    b.save()
    b = new("tube", 0.008)
    b.cup(0.008, 0.100, 0.001, 0.001, color=(0.65, 0.80, 0.85))
    b.save()
    b = new("rack")
    b.shape("box", (0, 0, 0.002), (0.102, 0.073, 0.004), BLUE)
    for x in (-0.03, 0.0, 0.03):
        for y in (-0.015, 0.015):
            b.ring(0.0103, 0.002, 0.046, 0.004, color=BLUE, segments=24, xy=(x, y))
    for x in (-0.047, 0.047):
        b.shape("box", (x, 0, 0.027), (0.008, 0.070, 0.046), BLUE)
    b.save()
    b = new("tube_stand")
    b.ring(0.0105, 0.002, 0.05, 0, color=BLUE)
    b.shape("box", (0, 0, 0.002), (0.04, 0.04, 0.004), BLUE)
    b.save()
    for name in ["source_vessel", "target_vessel", "tip_bin"]:
        b = new(name, 0.025 if name != "tip_bin" else None)
        color = {
            "source_vessel": (0.20, 0.60, 0.80),
            "target_vessel": (0.85, 0.45, 0.20),
            "tip_bin": (0.25, 0.30, 0.35),
        }[name]
        b.cup(0.008 if name != "tip_bin" else 0.023, 0.045, 0.002, 0.002, color=color)
        b.save()
    b = new("sample", 0.035)
    b.shape("cylinder", (0, 0, 0.035), (0.016, 0.07), BLUE)
    b.shape("cylinder", (0, 0, 0.073), (0.018, 0.006), WHITE)
    b.save()
    b = new("bolt", 0.035)
    b.shape("cylinder", (0, 0, 0.0175), (0.005, 0.035), STEEL)
    b.shape("cylinder", (0, 0, 0.0385), (0.0085, 0.007), STEEL)
    b.save()
    b = new("jig")
    b.shape("box", (0, 0, 0.003), (0.225, 0.16, 0.006), BLUE)
    for x in (-0.105, 0.105):
        b.shape("box", (x, 0, 0.012), (0.006, 0.14, 0.012), BLUE)
    for y in (-0.072, 0.072):
        b.shape("box", (0, y, 0.012), (0.204, 0.006, 0.012), BLUE)
    b.save()
    b = new("tip", 0.0005)
    b.cup(0.0035, 0.04, 0.0005, 0.0005, color=(0.7, 0.8, 0.82))
    b.save()
    b = new("tip_holder")
    for x in (0.0, 0.012, 0.024):
        for y in (0.0, 0.012):
            b.ring(0.0055, 0.0015, 0.012, color=BLUE, xy=(x, y))
    b.shape("box", (0.012, 0.006, 0.0015), (0.04, 0.028, 0.003), BLUE)
    b.save()
    b = new("pipette_holder")
    b.shape("box", (0, 0, 0.003), (0.07, 0.07, 0.006), BLUE)
    b.shape("box", (0, 0.025, 0.055), (0.012, 0.012, 0.104), BLUE)
    b.ring(0.014, 0.003, 0.012, 0.09, color=BLUE)
    b.save()
    b = new("pipette")
    body = UsdGeom.Xform.Define(b.stage, "/pipette/body").GetPrim()
    b.rigid(body, 0.085)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    b.shape("cylinder", (0, 0, 0.085), (0.011, 0.13), WHITE, body)
    b.shape("cylinder", (0, 0, 0.0125), (0.003, 0.025), BLUE, body)
    b.shape("box", (0, 0.012, 0.08), (0.016, 0.008, 0.06), BLUE, body)
    b.shape("cylinder", (0, 0, 0.025), (0.014, 0.006), BLUE, body)
    plunger = UsdGeom.Xform.Define(b.stage, "/pipette/plunger").GetPrim()
    b.rigid(plunger, 0.005)
    UsdGeom.Xformable(plunger).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.155))
    b.shape("cylinder", (0, 0, 0.005), (0.006, 0.01), BLUE, plunger)
    b.shape("cylinder", (0, 0, -0.004), (0.003, 0.012), STEEL, plunger)
    j = UsdPhysics.PrismaticJoint.Define(b.stage, "/pipette/plunger_joint")
    j.CreateBody0Rel().SetTargets([body.GetPath()])
    j.CreateBody1Rel().SetTargets([plunger.GetPath()])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.155))
    j.CreateLowerLimitAttr(-0.008)
    j.CreateUpperLimitAttr(0.0)
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(160.0)
    d.CreateDampingAttr(2.0)
    d.CreateMaxForceAttr(3.0)
    d.CreateTargetPositionAttr(0.0)
    b.save()
    b = new("fridge")
    body = UsdGeom.Xform.Define(b.stage, "/fridge/body").GetPrim()
    b.rigid(body, 30.0)
    UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(True)
    # Static chassis, passive independent dynamic door connected by a physical hinge.
    for x in (-0.291, 0.291):
        b.shape("box", (x, 0, 0.55), (0.018, 0.55, 1.1), WHITE, body)
    b.shape("box", (0, 0.267, 0.55), (0.57, 0.016, 1.1), WHITE, body)
    for z in (0.01, 0.40, 0.98, 1.09):
        b.shape("box", (0, 0, z), (0.57, 0.53, 0.02), WHITE, body)
    door = UsdGeom.Xform.Define(b.stage, "/fridge/door").GetPrim()
    b.rigid(door, 2.0)
    UsdGeom.Xformable(door).AddTranslateOp().Set(Gf.Vec3d(-0.30, -0.30, 0))
    b.shape("box", (0.30, 0, 0.55), (0.60, 0.026, 1.1), WHITE, door)
    b.shape("box", (0.52, -0.04, 0.65), (0.018, 0.018, 0.16), STEEL, door)
    for z in (0.57, 0.73):
        b.shape("box", (0.52, -0.022, z), (0.018, 0.04, 0.018), STEEL, door)
    j = UsdPhysics.RevoluteJoint.Define(b.stage, "/fridge/door_hinge")
    j.CreateBody1Rel().SetTargets([door.GetPath()])
    # Static body0 is a collision-only Xform; use the world anchor frame at runtime.
    j.CreateBody0Rel().SetTargets([body.GetPath()])
    j.CreateLocalPos0Attr(Gf.Vec3f(-0.30, -0.30, 0))
    j.CreateAxisAttr("Z")
    j.CreateLowerLimitAttr(-110.0)
    j.CreateUpperLimitAttr(0.0)
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(0.15)
    d.CreateMaxForceAttr(0.5)
    d.CreateTargetVelocityAttr(0.0)
    b.save()
    for p in destination.glob("*.usda"):
        s = Usd.Stage.Open(str(p))
        root = s.GetDefaultPrim()
        r = (
            UsdGeom.BBoxCache(0, ["default", "render"])
            .ComputeWorldBound(root)
            .ComputeAlignedRange()
        )
        manifest["assets"][p.stem] = {
            "file": p.name,
            "dimensions_m": list(r.GetSize()),
            "estimated_properties": True,
            "mass_kg": UsdPhysics.MassAPI(root).GetMassAttr().Get()
            if root.HasAPI(UsdPhysics.MassAPI)
            else None,
            "collision": "analytic hollow multi-primitive",
            "grasp_frame": {"xyz": [0, 0, float(r.GetSize()[2]) * 0.6]},
            "bounds_min": list(r.GetMin()),
            "bounds_max": list(r.GetMax()),
        }
    manifest["assets"]["lid"]["grasp_frame"] = {
        "xyz": [0.0, -0.050, 0.031],
        "purpose": "parallel pinch of handling clip",
    }
    manifest["assets"]["tube"]["grasp_frame"] = {
        "xyz": [0.0, 0.0, 0.065],
        "purpose": "side pinch above rack",
    }
    manifest["assets"]["pipette"]["grasp_frame"] = {
        "xyz": [0.0, 0.0, 0.085],
        "purpose": "body side pinch",
    }
    manifest["assets"]["fridge"]["handle_frame"] = {"xyz": [0.22, -0.34, 0.65]}
    manifest["assets"]["insert_clearance"].update(
        {
            "source": "HTR_Insert.stp visual; STEP-derived hollow collision proxy",
            "visual_diameter_mm": 19.75,
            "collision_outer_diameter_mm": 19.55,
            "collision_inner_diameter_mm": 12.25,
            "collision": "32 convex annular sectors plus analytic bottom; opening retained",
            "clearance_status": "estimated outer clearance; replace with measured fit",
        }
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
