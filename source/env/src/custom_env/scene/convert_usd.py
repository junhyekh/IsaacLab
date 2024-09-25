import argparse

from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("target_idx", type=int)
# parser.add_argument("end_idx", type=int)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.app
from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg

from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

from custom_env.util.path import ensure_directory
from custom_env.scene.dgn_object_set import DGNObjectSet

def _convert_usd(urdf: str, out_path: str):
    out_path = Path(out_path)
    k = Path(urdf).stem
    usd_path = ensure_directory(out_path / 'usd' / k)

    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf,
        usd_dir=usd_path,
        usd_file_name=f'{k}.usd',
        fix_base=False,
        convex_decompose_mesh=True,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        make_instanceable=True,
        self_collision=True
    )
    urdf_converter = UrdfConverter(urdf_converter_cfg) 
    # print("URDF importer output:")
    # print(f"Generated USD file: {urdf_converter.usd_path}")
    # print("-" * 80)
    # print("-" * 80)

def _convert_urdf_to_usd():
    # need:
    # keys

    dgn = DGNObjectSet(DGNObjectSet.Config(data_path='/input/DGN/meta-v8'))
    # names = names[:100]
    # no real "label" available for now...
    names = np.sort(np.asarray([dgn.urdf(key) for key in dgn.keys()]))
    print(names[:10])
    print(args_cli.target_idx, names[0])
    # urdf (OK?)
    out_path = ensure_directory(F'/input/DGN/meta-v8')

    # for name in tqdm(names):
    #     __convert_one(name, out_path)
    # convert_fn = partial(_convert_usd, out_path=str(out_path))
    # ctx = mp.get_context('spawn')
    # with ctx.Pool(16) as pool:
    #     for _ in tqdm(pool.imap_unordered(convert_fn, names),
    #                   total=len(names), desc='convert'):
    #         pass

    # for name in names[args_cli.target_idx]:
    _convert_usd(names[args_cli.target_idx], out_path)
        # print(name)

def main():
    _convert_urdf_to_usd()

if __name__ == "__main__":
    main()
    simulation_app.close()