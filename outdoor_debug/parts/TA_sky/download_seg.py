import sys, os
import tartanair as ta
root = sys.argv[1]
envs = sys.argv[2].split(",")
ta.init(root)
ta.download(env=envs, difficulty=["easy"], modality=["seg"], camera_name=["lcam_front"], unzip=False, num_workers=1, data_source=sys.argv[3] if len(sys.argv) > 3 else "huggingface")
