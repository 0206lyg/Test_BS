#!/usr/bin/env python3
"""Compare a pair of every-step histories at matching physical times."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('coarse', type=Path)
p.add_argument('fine', type=Path)
a = p.parse_args()
def load(path):
    return {round(float(r['time_s']), 14):r for r in csv.DictReader(path.open())}
c, f = load(a.coarse), load(a.fine)
times = sorted(c.keys() & f.keys())
out = {'coarse':str(a.coarse), 'fine':str(a.fine), 'matching_times':len(times),
       'last_time_s':times[-1] if times else None}
for field in ['eta_relative', 'sigma_contact_Pa', 'max_overlap_m',
              'normal_elastic_energy_J', 'tangential_elastic_energy_J', 'rolling_elastic_energy_J']:
    if not times or field not in c[times[0]] or field not in f[times[0]]:
        continue
    x = np.array([float(c[t][field]) for t in times])
    y = np.array([float(f[t][field]) for t in times])
    diff = x-y
    scale = max(float(np.max(np.abs(y))), 1e-300)
    out[field] = {'maximum_absolute_difference':float(np.max(np.abs(diff))),
                  'rms_difference_over_fine_max':float(np.sqrt(np.mean(diff**2))/scale),
                  'last_coarse':float(x[-1]), 'last_fine':float(y[-1]),
                  'last_difference_over_fine_max':float(diff[-1]/scale)}
print(json.dumps(out,indent=2))
