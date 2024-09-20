#!/usr/bin/env bash

set -exu

sudo ln -s /usr/bin/python3.10 /usr/bin/python
cd /home/user/IsaacLab
sudo chmod 777 -R /home/user/.cache

./isaaclab.sh --install