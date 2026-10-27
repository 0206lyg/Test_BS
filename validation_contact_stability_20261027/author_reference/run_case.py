from pathlib import Path
import sys,os,time,subprocess,json,resource
root=Path(__file__).parent;name=sys.argv[1];d=root/name
exe=os.environ.get('LF_DEM_EXE', 'LF_DEM')
cmd=[exe,'-f','-r','1h','../n500_ar1.dat','params.txt']
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_THREAD_LIMIT='1')
t=time.monotonic()
with (d/'console.log').open('w') as out:
 try:
  p=subprocess.run(cmd,cwd=d,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=240);status=p.returncode
 except subprocess.TimeoutExpired:status='timeout240'
r={'status':status,'wall_seconds':time.monotonic()-t,'maxrss_kb':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'command':cmd,'threads':1}
(d/'timing.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
