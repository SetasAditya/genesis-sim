"""Geometry for the reusable, top-only rack."""

from xml.etree import ElementTree as ET

from wetlab_isaaclab.assets import ASSET_ROOT


CROSSBAR_RACK_URDF = ASSET_ROOT / "equipment/crossbar_rack/pipette_crossbar_rack.urdf"


def crossbar_rack_boxes():
    """Return the authored rack's local collision-box centers and dimensions."""
    model = ET.parse(CROSSBAR_RACK_URDF)
    return {
        collider.get("name"): (
            tuple(map(float, collider.find("origin").get("xyz").split())),
            tuple(map(float, collider.find("geometry/box").get("size").split())),
        )
        for collider in model.findall("link/collision")
    }
