"""Native Isaac Sim sandbox. SimulationApp must exist before this module is imported."""

import json
import math
import random
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from .asset_builder import build_assets
from .config import (
    Action,
    ObjectState,
    BENCH_Z,
    PHYSICS_DT,
    CONTROL_DT,
    STATIONS,
    TASK_IDS,
    layout,
)
from .tasks import TaskEvaluator, VolumeModel, distance
from .robot import GalbotController, matrix


def reference(stage, path, asset, position=(0, 0, 0)):
    p = stage.DefinePrim(path, "Xform")
    p.GetReferences().AddReference(str(asset))
    x = UsdGeom.Xformable(p)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*position))
    return p


def rotation(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


class WetLabSandbox:
    def __init__(
        self,
        root,
        task="petri",
        mode="open",
        seed=0,
        render=True,
        insert_collision="clearance",
    ):
        if insert_collision not in ("nominal", "clearance"):
            raise ValueError("Unknown insert collision model")
        self.insert_collision = insert_collision
        self.root = Path(root)
        self.output = self.root / "outputs/sandbox"
        self.output.mkdir(parents=True, exist_ok=True)
        self.asset_dir = self.root / "assets/sandbox"
        self.render_enabled = render
        self.manifest = build_assets(self.asset_dir)
        self.context = omni.usd.get_context()
        self.context.new_stage()
        self.stage = self.context.get_stage()
        self.stage.SetDefaultPrim(self.stage.DefinePrim("/World", "Xform"))
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        self.lab = reference(
            self.stage,
            "/World/LabUtopia",
            self.root / "assets/generated/lab_collision.usda",
        )
        self.cleared_clutter = []
        for group in ("group_001", "group_002"):
            prim = self.stage.GetPrimAtPath("/World/LabUtopia/lab_001/" + group)
            if prim:
                UsdGeom.Imageable(prim).MakeInvisible()
                for child in Usd.PrimRange(prim):
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
                self.cleared_clutter.append(str(prim.GetPath()))
        self.galbot = reference(
            self.stage,
            "/World/Galbot",
            self.root / "assets/generated/galbot_standing.usda",
        )
        initial_x = STATIONS[task][0]
        initial_y = 1.60 if task == "fridge_retrieval" else 2.66
        gx = UsdGeom.Xformable(self.galbot)
        gx.ClearXformOpOrder()
        placement = Gf.Matrix4d(1.0)
        placement.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90.0))
        placement.SetTranslateOnly(Gf.Vec3d(initial_x, initial_y, 0.0))
        gx.AddTransformOp().Set(placement)
        for prim in Usd.PrimRange(self.galbot):
            if prim.GetName() in [f"wheel{i}_joint" for i in range(1, 5)]:
                UsdPhysics.DriveAPI.Get(prim, "angular").CreateMaxForceAttr(20.0)
            elif prim.GetName() in ["left_gripper_joint", "right_gripper_joint"]:
                UsdPhysics.DriveAPI.Get(prim, "angular").CreateMaxForceAttr(1.5)
        self.body_paths = {}
        self.origins = {}
        self.sizes = {}
        self.initial = layout()
        self.fixed_paths = []
        self.handles = {}
        self.attachments = {}
        self.events = []
        self.debug_used = False
        self._compose()
        self._mount_magnet()
        self._author_cameras()
        light = UsdLux.DomeLight.Define(self.stage, "/World/Lighting")
        light.CreateIntensityAttr(700.0)
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=PHYSICS_DT,
            rendering_dt=1 / 60,
            backend="numpy",
            device="cpu",
        )
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(
                    prim
                ).CreateEnableEnhancedDeterminismAttr(True)
        self.robot = self.world.scene.add(
            SingleArticulation(
                "/World/Galbot/OmniChassis/base_link",
                name="galbot",
                reset_xform_properties=False,
            )
        )
        self.pipette_art = self.world.scene.add(
            SingleArticulation(
                "/World/Equipment/pipette/body",
                name="pipette_art",
                reset_xform_properties=False,
            )
        )
        for name, path in self.body_paths.items():
            handle = SingleRigidPrim(path, name=name, reset_xform_properties=False)
            self.handles[name] = (
                handle if name == "plunger" else self.world.scene.add(handle)
            )
        self.world.reset()
        self.handles["plunger"].initialize()
        self.controller = GalbotController(self.stage, self.robot)
        self.home = self.robot.get_joint_positions().copy()
        self.volume = VolumeModel()
        self.seed = seed
        self.targets = self._targets()
        self.reset(seed=seed, task=task, mode=mode)
        self.kinematic_check = self.controller.check_kinematics()
        (self.output / "kinematics.json").write_text(
            json.dumps(self.kinematic_check, indent=2)
        )
        print("KINEMATICS", self.kinematic_check, flush=True)
        if not all(v["passed"] for v in self.kinematic_check.values()):
            raise RuntimeError("Galbot FK disagrees with measured PhysX link frames")

    def _spawn(self, name, asset, pos, dynamic=True, body_suffix=""):
        p = reference(self.stage, "/World/Equipment/" + name, asset, pos)
        body = self.stage.GetPrimAtPath(str(p.GetPath()) + body_suffix)
        bbox = (
            UsdGeom.BBoxCache(0, ["default", "render"])
            .ComputeUntransformedBound(p)
            .ComputeAlignedRange()
        )
        lo = bbox.GetMin()
        hi = bbox.GetMax()
        origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], float)
        if Path(asset).parent == self.asset_dir and name != "insert":
            origin = np.zeros(3)
        self.origins[name] = origin
        self.sizes[name] = np.array(hi - lo)
        self.initial[name] = tuple(pos)
        if dynamic:
            self.body_paths[name] = str(body.GetPath())
        else:
            self.fixed_paths.append(str(p.GetPath()))
        # Position native CAD at its bottom-centered semantic frame.
        if not body_suffix:
            UsdGeom.Xformable(p).GetOrderedXformOps()[0].Set(
                Gf.Vec3d(*(np.array(pos) - origin))
            )
        return p

    def _compose(self):
        initial = self.initial.copy()
        for n in [
            "dish",
            "lid",
            "tube",
            "source_vessel",
            "target_vessel",
            "tip_bin",
            "sample",
            "tip",
        ]:
            self._spawn(
                n, self.asset_dir / f"{n}.usda", initial[n], dynamic=n != "tip_bin"
            )
        for n in ["rack", "jig"]:
            self._spawn(n, self.asset_dir / f"{n}.usda", initial[n], False)
        self._spawn(
            "tube_stand",
            self.asset_dir / "tube_stand.usda",
            (initial["tube"][0], initial["tube"][1], BENCH_Z),
            False,
        )
        self._spawn(
            "tip_holder",
            self.asset_dir / "tip_holder.usda",
            (initial["tip"][0], initial["tip"][1], BENCH_Z),
            False,
        )
        self._spawn(
            "pipette_holder",
            self.asset_dir / "pipette_holder.usda",
            (initial["pipette"][0], initial["pipette"][1], BENCH_Z),
            False,
        )
        htr = {
            "basket": "HTR_Bottom_Basket",
            "well": "HTR_Well",
            "insert": "HTR_Insert",
            "cover": "HTR_Cover",
            "disassembly_plate": "HTR_Disassembly_Plate",
        }
        self.htr_names = htr
        for n, f in htr.items():
            asset = (
                self.asset_dir / "insert_clearance.usda"
                if n == "insert" and self.insert_collision == "clearance"
                else self.root / "assets/htr" / f"{f}.usda"
            )
            part = self._spawn(n, asset, initial[n])
            rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(part)
            rigid.CreateSolverPositionIterationCountAttr(64)
            rigid.CreateSolverVelocityIterationCountAttr(8)
            rigid.CreateMaxDepenetrationVelocityAttr(0.02)
            rigid.CreateLinearDampingAttr(1.0)
            rigid.CreateAngularDampingAttr(1.0)
            for mesh in Usd.PrimRange(part):
                if mesh.HasAPI(UsdPhysics.CollisionAPI):
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
                    collision.CreateContactOffsetAttr(
                        0.00002 if n in ("well", "insert") else 0.0001
                    )
                    collision.CreateRestOffsetAttr(0.0)
            if n in ("cover", "well") or (
                n == "insert" and self.insert_collision == "nominal"
            ):
                for mesh in Usd.PrimRange(part):
                    if mesh.IsA(UsdGeom.Mesh):
                        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(
                            "sdf"
                        )
                        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(
                            mesh
                        ).CreateSdfResolutionAttr(
                            1024 if n == "well" else 512 if n == "insert" else 256
                        )
        for i in range(4):
            bolt = self._spawn(
                "bolt" + str(i),
                self.asset_dir / "bolt.usda",
                (-3.20 + i * 0.027, 3.70, BENCH_Z),
            )
            rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(bolt)
            rigid.CreateMaxDepenetrationVelocityAttr(0.02)
            rigid.CreateLinearDampingAttr(1.0)
            rigid.CreateAngularDampingAttr(1.0)
        self.fridge_position = np.array(initial["fridge"])
        p = reference(
            self.stage,
            "/World/Equipment/fridge",
            self.asset_dir / "fridge.usda",
            initial["fridge"],
        )
        j = UsdPhysics.RevoluteJoint(
            self.stage.GetPrimAtPath(str(p.GetPath()) + "/door_hinge")
        )
        # Use an explicit fixed kinematic chassis, so both joint frames are local.
        j.CreateBody0Rel().SetTargets([Sdf.Path("/World/Equipment/fridge/body")])
        j.CreateLocalPos0Attr(Gf.Vec3f(-0.30, -0.30, 0.0))
        self.body_paths["fridge_door"] = str(p.GetPath()) + "/door"
        self.origins["fridge_door"] = np.zeros(3)
        self.sizes["fridge_door"] = np.array([0.6, 0.026, 1.1])
        self.initial["fridge_door"] = tuple(self.fridge_position + [-0.30, -0.30, 0.0])
        self._spawn(
            "pipette",
            self.asset_dir / "pipette.usda",
            initial["pipette"],
            body_suffix="/body",
        )
        self.origins["pipette"] = np.zeros(3)
        self.body_paths["plunger"] = "/World/Equipment/pipette/plunger"
        self.origins["plunger"] = np.zeros(3)
        self.sizes["plunger"] = np.array([0.012, 0.012, 0.02])
        self.initial["plunger"] = tuple(
            np.array(initial["pipette"]) + [0.0, 0.0, 0.155]
        )
        self._write_provenance()

    def _write_provenance(self):
        source = json.loads((self.root / "assets/htr/manifest.json").read_text())
        comparisons = {}
        pdf = {
            "basket": [0.1975, 0.123, 0.046],
            "cover": [0.1975, 0.123, 0.026],
            "well": [0.128, 0.085, 0.028],
        }
        for n, f in self.htr_names.items():
            comparisons[n] = {
                "source": f + ".stp",
                "cad_dimensions_m": self.sizes[n].tolist(),
                "mass_kg": source[f].get("mass_kg"),
                "pdf_dimensions_m": pdf.get(n),
                "mass_estimated": source[f].get("mass_kg") is None or n == "insert",
            }
            if n == "cover":
                comparisons[n]["dimension_note"] = (
                    "Bounding height includes underside bosses; preserve CAD rather than rescale to PDF plate thickness."
                )
            if n == "insert":
                comparisons[n]["fit_note"] = (
                    "Nominal insert and bore are both 19.75 mm. Clearance mode retains CAD visuals and uses an estimated 19.55 mm collider; measure real clearance before calibration."
                )
                comparisons[n]["collision_mode"] = self.insert_collision
                comparisons[n]["visual_diameter_mm"] = 19.75
                comparisons[n]["collision_diameter_mm"] = (
                    19.55 if self.insert_collision == "clearance" else 19.75
                )
                comparisons[n]["collision_clearance_estimated"] = (
                    self.insert_collision == "clearance"
                )
            comparisons[n]["estimated_contact_parameters"] = {
                "contact_offset_m": 0.00002 if n in ("well", "insert") else 0.0001,
                "rest_offset_m": 0.0,
                "max_depenetration_velocity_m_s": 0.02,
                "linear_damping_per_s": 1.0,
                "angular_damping_per_s": 1.0,
            }
            if n in pdf:
                comparisons[n]["cad_minus_pdf_m"] = (self.sizes[n] - pdf[n]).tolist()
        (self.asset_dir / "htr_audit.json").write_text(
            json.dumps(comparisons, indent=2)
        )

    def _targets(self):
        d = self.initial
        rack = np.array(d["rack"])
        jig = np.array(d["jig"])
        # HTR target heights derive from actual source mesh extents; use a pose audit
        # to verify nested seating rather than treating heights as assembly proof.
        basket = jig + np.array([0.0, 0.0, 0.006])
        well = basket + np.array([0.0, 0.0, 0.0092])
        insert = well + np.array([-0.045, -0.026, 0.0049])
        # CAD features: basket top 36.6 mm, bottom -9.2 mm; cover underside
        # bosses project 10.25 mm below its mating plane. Bottom-centered frames
        # therefore put the cover minimum 35.55 mm above the basket minimum.
        cover = basket + np.array([0.0, 0.0, 0.03555])
        targets = {
            "lid_park": d["lid_park"],
            "lid_seated": tuple(np.array(d["dish"]) + [0, 0, 0.005]),
            "tube_seated": tuple(rack + [-0.03, -0.015, 0.004]),
            "basket": tuple(basket),
            "well": tuple(well),
            "insert": tuple(insert),
            "cover": tuple(cover),
            "sample_target": d["sample_target"],
        }
        for i, (x, y) in enumerate(
            [(-0.080, -0.04555), (0.080, -0.04555), (-0.080, 0.04555), (0.080, 0.04555)]
        ):
            targets["bolt" + str(i)] = tuple(cover + [x, y, 0.00055])
        (self.asset_dir / "task_targets.json").write_text(json.dumps(targets, indent=2))
        return targets

    def _mount_magnet(self):
        wrist = str(
            next(
                p.GetPath()
                for p in Usd.PrimRange(self.galbot)
                if p.GetName() == "left_arm_link7"
            )
        )
        wp = self.stage.GetPrimAtPath(wrist)
        flange = next(p for p in Usd.PrimRange(wp) if p.GetName() == "gripper_flange")
        w = matrix(UsdGeom.Xformable(wp).ComputeLocalToWorldTransform(0))
        f = matrix(UsdGeom.Xformable(flange).ComputeLocalToWorldTransform(0))
        f[:3, 3] += f[:3, :3] @ np.array([0.14, 0, 0])
        pad = UsdGeom.Cylinder.Define(self.stage, wrist + "/MagneticPad")
        pad.CreateRadiusAttr(0.012)
        pad.CreateHeightAttr(0.006)
        pad.CreateAxisAttr("Z")
        m = np.linalg.inv(w) @ f
        m[:3, 3] += m[:3, 2] * 0.063
        UsdGeom.Xformable(pad).AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))
        pad.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.24, 0.28)])
        self.magnet_collision = UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        a = PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim())
        a.CreateContactOffsetAttr(0.0001)
        a.CreateRestOffsetAttr(0.0)
        post = UsdGeom.Cylinder.Define(self.stage, wrist + "/MagneticPost")
        post.CreateRadiusAttr(0.004)
        post.CreateHeightAttr(0.060)
        post.CreateAxisAttr("Z")
        m = np.linalg.inv(w) @ f
        m[:3, 3] += m[:3, 2] * 0.030
        UsdGeom.Xformable(post).AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))
        post.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.32, 0.34)])
        UsdPhysics.CollisionAPI.Apply(post.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(wp)
        mass.CreateMassAttr(float(mass.GetMassAttr().Get() or 0.01) + 0.05)

    def _author_cameras(self):
        cameras = {"overview": ((-3.8, -0.65, 2.4), (-3.85, 2.8, 0.85), 14.0)}
        for name, (x, y) in STATIONS.items():
            eye = (
                (x + 2.0, 3.15, 1.85)
                if name != "fridge_retrieval"
                else (x + 1.7, 1.2, 1.75)
            )
            cameras[name] = (eye, (x, y + 0.05, 0.85), 16.0)
        for name, (eye, target, focal) in cameras.items():
            camera = UsdGeom.Camera.Define(self.stage, "/World/Cameras/" + name)
            camera.CreateFocalLengthAttr(focal)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
            m = (
                Gf.Matrix4d()
                .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
                .GetInverse()
            )
            UsdGeom.Xformable(camera).AddTransformOp().Set(m)
        self.stage.GetRootLayer().customLayerData = {
            "default_camera": "/World/Cameras/overview"
        }

    def check_openings(self):
        from omni.physx import get_physx_scene_query_interface

        query = get_physx_scene_query_interface()
        checks = []
        x, y, _ = self.initial["rack"]
        for dx in (-0.03, 0.0, 0.03):
            for dy in (-0.015, 0.015):
                hit = query.raycast_closest(
                    (x + dx, y + dy, BENCH_Z + 0.06), (0.0, 0.0, -1.0), 0.065
                )
                checks.append(
                    {
                        "position": [x + dx, y + dy],
                        "hit_distance_m": float(hit.get("distance", 0.0)),
                        "passed": bool(
                            hit.get("hit") and hit.get("distance", 0.0) > 0.05
                        ),
                    }
                )
        return checks

    def _magnet_contact(self, name, tcp):
        from omni.physx import get_physx_scene_query_interface

        direction = tcp[:3, 2]
        origin = tcp[:3, 3] + direction * 0.0001
        hit = get_physx_scene_query_interface().raycast_closest(
            tuple(origin), tuple(direction), 0.002
        )
        return bool(
            hit.get("hit")
            and str(hit.get("rigidBody", "")).startswith(self.body_paths[name])
        )

    def _pose(self, name):
        p, q = self.handles[name].get_world_pose()
        return np.array(p) + rotation(q).apply(self.origins[name]), np.array(q)

    def _set_pose(self, name, position, orientation=(1.0, 0.0, 0.0, 0.0)):
        q = np.array(orientation)
        p = np.array(position) - rotation(q).apply(self.origins[name])
        self.handles[name].set_world_pose(position=p, orientation=q)
        self.handles[name].set_linear_velocity(np.zeros(3))
        self.handles[name].set_angular_velocity(np.zeros(3))

    def _remove_constraints(self):
        for path in list(self.attachments):
            self.stage.RemovePrim("/World/Constraints/" + path)
        self.attachments.clear()
        if self.stage.GetPrimAtPath("/World/Constraints/basket_clamp"):
            self.stage.RemovePrim("/World/Constraints/basket_clamp")

    def reset(self, seed=0, task=None, mode="open", jitter_m=0.0):
        if task is not None and task not in TASK_IDS:
            raise ValueError(f"Unknown task {task}")
        self.task = task or self.task
        self.seed = seed
        self.mode = mode
        self._remove_constraints()
        self.volume.reset()
        self.events = []
        self.debug_used = False
        rng = random.Random(seed)
        for n in self.handles:
            if n == "plunger":
                continue
            p = np.array(self.initial[n])
            if n in (
                "lid",
                "tube",
                "basket",
                "well",
                "insert",
                "cover",
                "sample",
                "pipette",
            ):
                p[:2] += [
                    rng.uniform(-jitter_m, jitter_m),
                    rng.uniform(-jitter_m, jitter_m),
                ]
            self._set_pose(n, p)
        self.pipette_art.set_joint_positions(np.zeros(self.pipette_art.num_dof))
        self.pipette_art.set_joint_velocities(np.zeros(self.pipette_art.num_dof))
        # Initial support surfaces: tube in stand; dish lid and tip placed on fixtures.
        self._set_pose(
            "tube", (self.initial["tube"][0], self.initial["tube"][1], BENCH_Z + 0.004)
        )
        if mode == "close":
            self._set_pose("lid", self.targets["lid_park"])
        x, y = STATIONS[self.task]
        if self.task == "fridge_retrieval":
            base = (x, 1.60, 0.075)
        else:
            base = (x, 2.66, 0.075)
        self.robot.set_world_pose(
            position=np.array(base),
            orientation=np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]),
        )
        self.robot.set_joint_positions(self.home)
        self.robot.set_joint_velocities(np.zeros_like(self.home))
        self.controller.targets = self.home.copy()
        self.controller.wrist_goals.clear()
        self.controller.gripper_goals.clear()
        self.controller.base_anchor = None
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))
        self.magnetic_enabled = False
        self.tip_attached = False
        self.tip_seen = False
        self.steps = 0
        self.evaluator = TaskEvaluator(self.task, self.targets, mode=mode)
        for _ in range(120):
            self.world.step(render=False)
        self.evaluator.reset()
        print("RESET_READY", self.task, seed, flush=True)
        return self.observe(), {"task": self.task, "seed": seed, "mode": mode}

    def object_states(self):
        out = {}
        for n, h in self.handles.items():
            p, q = self._pose(n)
            v = h.get_linear_velocity()
            w = h.get_angular_velocity()
            out[n] = ObjectState(
                tuple(float(x) for x in p),
                tuple(float(x) for x in q),
                float(np.linalg.norm(v)),
                float(np.linalg.norm(w)),
                float(rotation(q).as_matrix()[2, 2]),
            )
        return out

    def observe(self):
        s = self.object_states()
        base, rot = self.robot.get_world_pose()
        return {
            "task": self.task,
            "seed": self.seed,
            "objects": {n: vars(v) for n, v in s.items()},
            "robot": {
                "base_position": base.tolist(),
                "base_orientation": rot.tolist(),
                "joint_names": self.controller.names,
                "joint_positions": self.robot.get_joint_positions().tolist(),
                "tcp": {
                    side: self.controller.tcp(side)[:3, 3].tolist()
                    for side in ("left", "right")
                },
            },
            "tools": {
                "magnetic_enabled": self.magnetic_enabled,
                "attached": list(self.attachments),
                "tip_attached": self.tip_attached,
            },
            "pipette_volume": self.volume.observation(),
            "task_state": self.evaluator.result(),
            "debug_used": self.debug_used,
        }

    def _joint(self, name, body0, body1, break_force=150.0):
        path = "/World/Constraints/" + name
        joint = UsdPhysics.FixedJoint.Define(self.stage, path)
        if body0:
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        joint.CreateExcludeFromArticulationAttr(True)

        def actual_world(path):
            if path in self.controller.wrists.values():
                return Gf.Matrix4d(self.controller.link_world(path).T.tolist())
            name = next((n for n, p in self.body_paths.items() if p == path), None)
            if name is not None:
                p, q = self.handles[name].get_world_pose()
                m = Gf.Matrix4d(1.0)
                m.SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*(float(v) for v in q[1:]))))
                m.SetTranslateOnly(Gf.Vec3d(*(float(v) for v in p)))
                return m
            return UsdGeom.Xformable(
                self.stage.GetPrimAtPath(path)
            ).ComputeLocalToWorldTransform(0)

        w1 = actual_world(body1)
        w0 = actual_world(body0) if body0 else Gf.Matrix4d(1)
        local = w1 * w0.GetInverse()
        joint.CreateLocalPos0Attr(Gf.Vec3f(local.ExtractTranslation()))
        joint.CreateLocalRot0Attr(Gf.Quatf(local.ExtractRotationQuat()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateBreakForceAttr(break_force)
        joint.CreateBreakTorqueAttr(10.0)
        return path

    def _tool_step(self):
        # A virtual magnetic pad is mounted at the left gripper TCP. Attachment
        # requires steel, a 2mm face gap, and pad alignment; it never changes poses.
        if self.magnetic_enabled and not any(
            n in self.attachments for n in ("basket", "cover")
        ):
            tcp = self.controller.magnetic_tcp()
            z = tcp[:3, 2]
            for n in ("basket", "cover"):
                p, q = self._pose(n)
                z_face = 0.0092 if n == "basket" else self.sizes[n][2]
                face = p + rotation(q).apply([0, 0, z_face])
                if (
                    distance(tcp[:3, 3], face) <= 0.002
                    and abs(z @ rotation(q).apply([0, 0, 1]))
                    >= math.cos(math.radians(10))
                    and self._magnet_contact(n, tcp)
                ):
                    self._joint(n, self.controller.wrists["left"], self.body_paths[n])
                    self.attachments[n] = "magnetic"
                    self.events.append(
                        {"event": "magnetic_attached", "object": n, "step": self.steps}
                    )
        if not self.magnetic_enabled:
            for n in list(self.attachments):
                if self.attachments[n] == "magnetic":
                    self.stage.RemovePrim("/World/Constraints/" + n)
                    del self.attachments[n]
        for n in list(self.attachments):
            p = self.stage.GetPrimAtPath("/World/Constraints/" + n)
            if not p or p.GetAttribute("physics:jointEnabled").Get() is False:
                del self.attachments[n]
                if n == "tip":
                    self.tip_attached = False
        # Tip friction-fit: axial engagement must occur at the pipette nozzle.
        pp, pq = self._pose("pipette")
        tp, tq = self._pose("tip")
        tip_mount = tp + rotation(tq).apply([0, 0, 0.04])
        if (
            not self.tip_attached
            and not self.volume.ejected
            and distance(pp, tip_mount) < 0.0015
            and rotation(pq).as_matrix()[2, 2] > 0.98
            and rotation(pq).as_matrix()[:, 2] @ rotation(tq).as_matrix()[:, 2] > 0.985
        ):
            self._joint("tip", self.body_paths["pipette"], self.body_paths["tip"], 15.0)
            self.attachments["tip"] = "tip_fit"
            self.tip_attached = True
            self.tip_seen = True
        # Eject only at full plunger overtravel and above the disposal opening.
        stroke = max(
            0.0, min(1.0, -float(self.pipette_art.get_joint_positions()[0]) / 0.008)
        )
        binpos = np.array(self.initial["tip_bin"])
        over_bin = distance(pp[:2], binpos[:2]) < 0.018 and pp[2] > binpos[2] + 0.045
        if self.tip_attached and stroke > 0.98 and over_bin:
            self.stage.RemovePrim("/World/Constraints/tip")
            self.attachments.pop("tip", None)
            self.tip_attached = False
            self.volume.eject(True)
        tip_end = tp
        region = None
        for n, tag in [("source_vessel", "source"), ("target_vessel", "target")]:
            vp, _ = self._pose(n)
            liquid_top = 0.002 + self.volume.source * 1e-9 / (math.pi * 0.006**2)
            upper = liquid_top if tag == "source" else 0.042
            if (
                distance(tip_end[:2], vp[:2]) < 0.005
                and vp[2] + 0.0025 < tip_end[2] < vp[2] + upper
            ):
                region = tag
        self.volume.update(stroke, self.tip_attached, region)

    def step(self, action):
        if action is None:
            action = Action()
        if isinstance(action, dict):
            action = Action(**action)
        if not isinstance(action, Action):
            raise TypeError("Expected Action or command dict")
        self.controller.command(action, CONTROL_DT)
        if action.magnetic is not None:
            self.magnetic_enabled = bool(action.magnetic)
        if action.fixture_clamp:
            p, q = self._pose("basket")
            if (
                distance(p, self.targets["basket"]) < 0.003
                and self.handles["basket"]
                .get_linear_velocity()
                .dot(self.handles["basket"].get_linear_velocity())
                < 0.0004
            ):
                self._joint("basket_clamp", None, self.body_paths["basket"], 300.0)
        elif action.fixture_clamp is False and self.stage.GetPrimAtPath(
            "/World/Constraints/basket_clamp"
        ):
            self.stage.RemovePrim("/World/Constraints/basket_clamp")
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            self.world.step(render=False)
        self.steps += 1
        self._tool_step()
        state = self.object_states()
        dq = rotation(state["fridge_door"].orientation).as_euler("xyz")[2]
        attached = list(self.attachments)
        # Conservatively suppress release scoring near a closed hand. This guard
        # is not a contact sensor and never creates a parallel-gripper weld.
        q = self.robot.get_joint_positions()
        for side in ("left", "right"):
            if q[self.controller.names.index(side + "_gripper_joint")] > 0.2:
                tcp = self.controller.tcp(side)[:3, 3]
                for n, o in state.items():
                    center = np.array(o.position) + [0, 0, self.sizes[n][2] / 2]
                    if distance(tcp, center) < 0.035:
                        attached.append(n)
        self.evaluator.update(
            state,
            CONTROL_DT,
            attached,
            {"fridge_door": abs(dq)},
            self.volume.observation(),
        )
        base_position, base_orientation = self.robot.get_world_pose()
        if (
            not np.isfinite(base_position).all()
            or not np.isfinite(base_orientation).all()
        ):
            self.evaluator.failure_reason = "robot_non_finite_state"
        elif (
            rotation(base_orientation).as_matrix()[2, 2] < math.cos(0.30)
            or base_position[2] > 0.25
        ):
            self.evaluator.failure_reason = "robot_lost_stable_base_support"
        obs = self.observe()
        info = obs["task_state"] | {"debug_used": self.debug_used}
        truncated = self.steps >= 18000
        if truncated and not info["success"]:
            info["failure_reason"] = "timeout"
        terminated = bool(info["success"] or info["failure_reason"])
        if self.render_enabled:
            self.world.render()
        return obs, 0.0, terminated, truncated, info

    def debug_set_pose(self, name, position):
        self.debug_used = True
        self._set_pose(name, position)

    def save(self):
        self.stage.GetRootLayer().Export(str(self.output / "all_tasks.usda"))
        return self.output / "all_tasks.usda"

    def close(self):
        self._remove_constraints()
        self.world.stop()
