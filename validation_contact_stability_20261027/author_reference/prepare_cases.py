from pathlib import Path
import json
root=Path(__file__).parent
meta=json.loads((root/'mapping.json').read_text())
base='''integration_method = 1;
friction_model = 1;
mu_static = 10;
mu_dynamic = 10;
mu_rolling = 10;
lubrication_model = tangential;
lub_reduce_parameter = 0.001;
lub_max_gap = 0.5;
contact_relaxation_time = -1h;
contact_relaxation_time_tan = -1h;
fixed_dt = {fixed};
dt = {dt}h;
dt_max = {dtmax}h;
disp_max = 0.0001;
kn = {kn}h;
kt = 0.8kn;
kr = 0.8kn;
auto_determine_knkt = false;
time_end = {strain}h;
time_interval_output_data = {interval}h;
time_interval_output_config = {strain}h;
out_data_particle = false;
out_data_interaction = false;
out_binary_conf = false;
'''
cases=[('original_fixed',meta['dimensionless_kn'],'true',.0001,.0001,.02,.0001),('original_adaptive',meta['dimensionless_kn'],'false',.0001,.0001,.02,.0001),('soft2000',2000,'false',.0001,.0001,.1,.001),('soft4000',4000,'false',.0001,.0001,.1,.001),('soft8000',8000,'false',.00005,.00005,.1,.001)]
for name,kn,fixed,dt,dtmax,strain,interval in cases:
 d=root/name;d.mkdir(exist_ok=True)
 (d/'params.txt').write_text(base.format(kn=kn,fixed=fixed,dt=dt,dtmax=dtmax,strain=strain,interval=interval))
