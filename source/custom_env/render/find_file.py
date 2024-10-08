import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ep_id", type=int)
parser.add_argument("--ep_ids", type=str)
parser.add_argument("--step", type=int)

args_cli = parser.parse_args()

import numpy as np
import pickle

def extract_number(file_name):
    return int(''.join(filter(str.isdigit, file_name.stem)))

def main():
    trajs = sorted(Path("/input/traj/route-2024-08-31/route/").glob("*.pkl"),
                   key=extract_number)
    if args_cli.ep_ids is None:
        indices = [1,2,3,6,7,8]
        for idx, fi in enumerate(trajs[3:7]):
            d = pickle.load(open(fi, "rb"))
            # if np.count_nonzero(d['episode_id']==args_cli.ep_id)>0:
            #     break
            is_bin = False
            # print(d['box_dim'][0])
            for k in range(len(d['episode_id'])):   
                height = d['box_pose'][k][1,2]
                # print(height,d['box_pose'][k][indices][:,2] )
                if ((d['box_pose'][k][-5:-1,2]> d['box_pose'][k][0,2]).all()
                    and (d['box_pose'][k][indices][:,2]==height).all()):
                    # d['box_pose'][k][:][2]
                    print(height,d['box_pose'][k][indices][:] )
                    # is_bin = True
                    # break
                    print(fi, k)
            # if is_bin:
            #     break
        # print(str(trajs[idx+args_cli.step]))
        print(fi, k)
    else:
        data = None
        ids = np.genfromtxt(args_cli.ep_ids, delimiter=',')
        for id in ids:
            for idx, fi in enumerate(trajs):
                d = pickle.load(open(fi, "rb"))
                if np.count_nonzero(d['episode_id']==id)>0:
                    break
            env_idx = np.flatnonzero(d['episode_id']==id)[0]
            if data is None:
                data = {}
                for k, v in d.items():
                    data[k] = [v[env_idx]]
            else:
                for k, v in d.items():
                    vv = data[k]
                    vv.append(v[env_idx])
                    data[k] = vv
        for k,v in data.items():
            data[k] = np.stack(v, 0)
        with open('/input/traj/stitched.pkl', 'wb') as fp:
            pickle.dump(data, fp)

if __name__ == "__main__":
    main()