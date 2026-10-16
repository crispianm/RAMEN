import sys, yaml, json
from pathlib import Path
sys.path.insert(0, '.')
import train
out = Path('outdoor_debug/parts/C2_config_train')
for name, f in [('c3_seed45','dn01_v3_bfix_C3_B'),('ramen_n46','dn01_c3rep_20261009_c3_n46'),('ft_n46','dn01_c3rep_20261009_ft_n46')]:
    cfg = train.load_config(Path(f'configs/experiments/{f}.yaml'))
    (out/f'effective_{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(name, 'ok')
