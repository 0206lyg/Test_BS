#!/usr/bin/env python3
"""Independent count/volume/size and coarse spatial-mixing audit of a LAMMPS fixture."""
import argparse
import cmath
import collections
import json
import math
from pathlib import Path


def audit(directory):
    directory=Path(directory)
    lines=(directory/'particles.data').read_text().splitlines()
    positions={};axes={};box={};section=None
    for line in lines:
        s=line.split()
        if not s:continue
        if s[0] in ('Atoms','Ellipsoids','Velocities'):
            section=s[0];continue
        if len(s)==4 and s[2] in ('xlo','ylo','zlo'):
            box[s[2][0]]=float(s[1])-float(s[0]);continue
        if section=='Atoms' and len(s)>=7:
            positions[int(s[0])]=tuple(map(float,s[4:7]))
        elif section=='Ellipsoids' and len(s)==8:
            axes[int(s[0])]=tuple(float(v)/2 for v in s[1:4])
    assert positions.keys()==axes.keys()
    volume={i:4/3*math.pi*a[0]*a[1]*a[2] for i,a in axes.items()}
    sizes=sorted(set(round((a[0]*a[1]*a[2])**(1/3),16) for a in axes.values()))
    groups={k:[] for k in range(len(sizes))}
    for i,a in axes.items():
        radius=(a[0]*a[1]*a[2])**(1/3)
        group=min(range(len(sizes)),key=lambda k:abs(sizes[k]-radius));groups[group].append(i)
    total=sum(volume.values());records=[]
    for k,ids in groups.items():
        octants=collections.Counter(tuple(int(positions[i][j]/box[axis]>=.5) for j,axis in enumerate('xyz')) for i in ids)
        fourier=[abs(sum(cmath.exp(2j*math.pi*positions[i][j]/box[axis]) for i in ids)/len(ids)) for j,axis in enumerate('xyz')]
        records.append({'species':k,'count':len(ids),'volume_equivalent_radius_m':sizes[k],
            'fraction_of_solid_volume':sum(volume[i] for i in ids)/total,
            'octant_counts':{''.join(map(str,key)):value for key,value in sorted(octants.items())},
            'axis_first_fourier_amplitudes':fourier})
    metadata=json.loads((directory/'parameters.json').read_text())
    expected_volume=len(axes)*math.pi/6*metadata['nominal_diameter_m']**3
    report={'count':len(axes),'species':records,'actual_phi':total/math.prod(box.values()),
        'total_particle_volume_m3':total,'nominal_total_volume_m3':expected_volume,
        'mean_volume_relative_error':abs(total/expected_volume-1),
        'aspect_ratio_range':[min(a[0]/a[2] for a in axes.values()),max(a[0]/a[2] for a in axes.values())],
        'size_ratio':sizes[-1]/sizes[0],
        'mixing_note':'Octants and Fourier amplitudes are descriptive diagnostics, not certification of equilibrium or absence of ordering.'}
    assert math.isclose(report['actual_phi'],metadata['phi'],rel_tol=1e-10)
    assert report['mean_volume_relative_error']<1e-10
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    r=audit(a.directory);(a.directory/'independent_fixture_audit.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
