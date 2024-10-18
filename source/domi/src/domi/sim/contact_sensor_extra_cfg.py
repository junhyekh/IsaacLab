#!/usr/bin/env python3

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import (
    ContactSensorCfg
)
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


from domi.sim.contact_sensor_extra import ContactSensorExtra


# NOTE(ycho): Visualize contact as arrow.
CONTACT_SENSOR_MARKER_CFG_2 = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)),
        )})


@configclass
class ContactSensorExtraCfg(ContactSensorCfg):
    class_type: type = ContactSensorExtra
    # hmm...
    max_contact_data_count: int = 0

    visualizer_cfg_2: VisualizationMarkersCfg = CONTACT_SENSOR_MARKER_CFG_2.replace(
        prim_path="/Visuals/ContactSensor_2")
    
    # Force threshold to visualize
    visualize_threshold: float = 0.1
