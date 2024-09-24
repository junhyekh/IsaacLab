import trimesh
from typing import Union, Optional
from pathlib import Path
import argparse
import numpy as np
from tqdm.auto import tqdm
import subprocess

URDF_TEMPLATE: str = '''<robot name="robot">
    <link name="base_link">
        <inertial>
            <mass value="{mass}"/>
            <origin xyz="{com}"/>
            <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}"
            iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{filename}" scale="{sx} {sy} {sz}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{collision_filename}" scale="{sx} {sy} {sz}"/>
            </geometry>
        </collision>
    </link>
</robot>
'''

def write_urdf(col : Union[str,Path],
                 vis : Optional[Union[str,Path]] = None,
                 out_path : Optional[Union[str,Path]] = None, 
                 scale : float = 1.):
    
    if isinstance(col, str):
        col = Path(col)

    scale = float(scale)
    print(col)
    new_obj = trimesh.load(col)
    new_obj.apply_scale(scale)

    volume = new_obj.volume
    # FIXME(ycho): fixed density
    density: float = 300.0
    mass = volume * density
    # trimesh moment of inertia is calculated
    # by taking the volume as the mass
    inertia = (mass / volume) * new_obj.moment_inertia
    if vis is None:
        vis = col
    elif isinstance(vis, str):
        vis = Path(vis)
    urdf_text = URDF_TEMPLATE.format(
        mass=mass,
        com=''.join([F' {x:.04f}' for x in new_obj.center_mass]),
        ixx=inertia[0, 0],
        ixy=inertia[0, 1],
        ixz=inertia[0, 2],
        iyy=inertia[1, 1],
        iyz=inertia[1, 2],
        izz=inertia[2, 2],
        filename=str(vis),
        collision_filename=str(col),
        sx=scale,
        sy=scale,
        sz=scale,
    )
    if out_path is None:
        out_path = vis.parent
    urdf_file = F'{out_path}/{vis.stem}.urdf'
    with open(str(urdf_file), 'w') as fp:
        fp.write(urdf_text)
    return urdf_file

def multiples(col, scale=0.1):
    out_path = '/tmp/DGN'
    p = list(Path(col).rglob("*.obj"))
    unique = []
    keys = []
    for pp in p:
        key = pp.stem.split('-')[1]
        if key in keys:
            continue
        else:
            print(key)
            keys.append(key)
            unique.append(pp)
    p = np.random.choice(unique, 30, False)
    for pp in tqdm(p):
        out = write_urdf(pp, out_path=out_path, scale=scale)
        # name = Path(out).stem
        # subprocess.run(f"python source/standalone/tools/convert_urdf.py \
        #                 {out} \
        #                /home/user/source/extensions/omni.isaac.lab_assets/data/DGN/{name}.usd \
        #                --make-instanceable --headless", shell=True)

def main(col, vis, out, scale):
    # write_urdf(col, vis, out, scale)
    multiples(col)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Utility to convert a mesh file into URDF format.")
    parser.add_argument("col", type=str, help="The path to the input collision mesh file.")
    parser.add_argument("-v", "--vis", type=str,
                        default=None,
                        help="The path to the input visual mesh file.")
    parser.add_argument("-o", "--output", type=str,
                        default=None,
                        help="The path to store the URDF file.")
    parser.add_argument(
        "-s,", "--scale",
        type=float,
        default=1.0,
    )
    args_cli = parser.parse_args()
    main(args_cli.col, args_cli.vis, args_cli.output, args_cli.scale)