from pathlib import Path
import json,numpy as np
root=Path(__file__).parent
names=['stiff_dt1e6s','stiff_dt1e7s','stiff_dt1e8s','stiff_dt1e8s_sparse','original_adaptive','soft2000','soft4000','soft8000','soft4000_halfdt']
summary={}
for name in names:
 d=root/name;files=list(d.glob('data*.dat'))
 if not files:continue
 a=np.loadtxt(files[0]);a=np.atleast_2d(a)
 tfile=d/'timing.json';timing=json.loads(tfile.read_text()) if tfile.exists() else {'status':'running'}
 y=a[a[:,1]>.001,4];dy=np.diff(y)
 summary[name]={**timing,'nout':len(a),'strain_end':a[-1,1],'eta_end':a[-1,3],'eta_min':a[:,3].min(),'eta_max':a[:,3].max(),'max_overlap_over_a':max(0.,-a[:,14].min()),'max_tangential_displacement_over_a':a[:,15].max(),'max_rolling_displacement_over_a':a[:,16].max(),'max_omega_over_gamma':a[:,21].max(),'output_dt_strain_min':a[a[:,22]>0,22].min(),'contact_eta_increment_lag1corr':float(np.corrcoef(dy[:-1],dy[1:])[0,1]) if len(dy)>2 else None,'contact_eta_increment_signflip_fraction':float(np.mean(dy[:-1]*dy[1:]<0)) if len(dy)>2 else None}
for base,ref in [('soft4000','soft4000_halfdt'),('soft2000','soft8000'),('soft4000','soft8000')]:
 a=np.loadtxt(next((root/base).glob('data*.dat')));b=np.loadtxt(next((root/ref).glob('data*.dat')))
 if a.shape==b.shape and np.allclose(a[:,1],b[:,1]):
  summary[f'{base}_vs_{ref}']={}
  for c,name in [(3,'eta'),(4,'contact_eta'),(14,'gap'),(16,'roll'),(21,'max_omega')]:
   summary[f'{base}_vs_{ref}'][name]={'max_abs_diff':float(np.max(np.abs(a[:,c]-b[:,c]))),'max_abs_diff_over_ref_max':float(np.max(np.abs(a[:,c]-b[:,c]))/max(np.max(np.abs(b[:,c])),1e-30))}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
