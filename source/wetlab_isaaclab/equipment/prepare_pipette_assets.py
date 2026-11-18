"""Convert the cabinet task's source descriptions for Isaac Sim 6.1."""

from copy import deepcopy
from pathlib import Path
from shutil import copytree
from xml.etree import ElementTree as ET

from wetlab_isaaclab.assets import ASSET_ROOT


def _stage_dg5f_sources(directory):
    import trimesh

    original = ASSET_ROOT / "robots/dexmate_dg5f"
    copytree(original, directory)
    pieces = {}
    for filename in ("whole_robot.urdf", "simulation.urdf"):
        path = directory / filename
        model = ET.parse(path)
        for link in model.findall("link"):
            name = link.get("name")
            if filename == "simulation.urdf" and (name == "root" or name.startswith("root_arm_1_link_")):
                # PhysX needs a rigid root and positive inertias for the virtual chain.
                # Use the same inertial placeholder as the source's base_link.
                link.find("inertial/mass").set("value", "0.001")
                link.find("inertial/inertia").attrib.update(ixx="1e-6", iyy="1e-6", izz="1e-6")
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is None or not mesh.get("filename").lower().endswith(".ply"):
                    continue
                relative = Path(mesh.get("filename"))
                if relative not in pieces:
                    geometry = trimesh.load_mesh(original / relative, process=False)
                    pieces[relative] = []
                    for index, part in enumerate(geometry.split(only_watertight=False)):
                        name = relative.with_name(f"{relative.stem}_piece_{index:02d}.obj")
                        (directory / name).write_text(trimesh.exchange.obj.export_obj(part, digits=12))
                        pieces[relative].append(name.as_posix())
                index = list(link).index(collision)
                link.remove(collision)
                for offset, name in enumerate(pieces[relative]):
                    piece = deepcopy(collision)
                    piece.find("geometry/mesh").set("filename", name)
                    link.insert(index + offset, piece)
        model.write(path, encoding="utf-8", xml_declaration=True)


def prepare_pipette_assets(output: Path) -> dict[str, Path]:
    """Create a new asset directory, including robot pair exclusions before cloning."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from pxr import Usd, UsdPhysics

    from wetlab_isaaclab.robots.dexmate import VEGA_1_URDF, author_vega_1_collision_filters
    from wetlab_isaaclab.robots.dg5f import author_dg5f_collision_filters

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    staged = output / "dg5f_source"
    _stage_dg5f_sources(staged)
    equipment = ASSET_ROOT / "equipment"
    sources = {
        "mounted": (staged / "whole_robot.urdf", True),
        "floating": (staged / "simulation.urdf", True),
        "vega": (VEGA_1_URDF, True),
        "pipette": (equipment / "gilson_pipette/pipette_gilson_articulated.urdf", False),
        "cabinet": (equipment / "cabinet/cabinet.urdf", False),
        "tube": (equipment / "test_tube/test_tube.urdf", False),
    }
    assets = {}
    for name, (source, anchored) in sources.items():
        converted = UrdfConverter(UrdfConverterCfg(
            asset_path=str(source), usd_dir=str(output / name),
            fix_base=anchored, merge_fixed_joints=False,
            force_usd_conversion=True, joint_drive=None,
        ))
        path = Path(converted.usd_path)
        stage = Usd.Stage.Open(str(path))
        root = stage.GetDefaultPrim()
        if name in ("mounted", "floating"):
            count = author_dg5f_collision_filters(stage, str(root.GetPath()), floating=name == "floating")
            if count != {"mounted": 224, "floating": 43}[name]:
                raise ValueError(f"{name}: unexpected collision exclusion count {count}")
        elif name == "vega":
            author_vega_1_collision_filters(stage, str(root.GetPath()))
        elif name in ("cabinet", "tube"):
            # A one-link fixture may import as geometry without a dynamic body.
            # Make its outer prim the single kinematic body for indexed reset.
            for prim in Usd.PrimRange(root):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if prim != root:
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
        stage.GetRootLayer().Save()
        assets[name] = path
    return assets
