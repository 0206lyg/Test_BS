#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np

path=Path(__file__).resolve().parents[2]/'diagnose_flake.py'
spec=importlib.util.spec_from_file_location('diagnose_flake',path)
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)

class LayerDiagnosticsTest(unittest.TestCase):
    def test_layers_random_alignment_separate(self):
        rng=np.random.default_rng(5);n=512;H=np.eye(3)
        modes,_=d.density_modes(H,np.array([0.,0.,1.]),.04,max_index=32,exact_gradient=True)
        random=rng.random((n,3));layered=random.copy();layered[:,2]=(np.arange(n)%8)/8
        null=d.UniformNull(32)
        a=d.density_metric(random,H,modes,null,'z');b=d.density_metric(layered,H,modes,null,'z')
        self.assertLess(a['z_peak_order'],.15)
        self.assertGreater(b['z_peak_order'],.999)
        self.assertLess(a['z_uniform_max_order_p95'],.2)
        S,_,_=d.nematic(np.tile([0.,0.,1.],(n,1)))
        self.assertAlmostEqual(S,1.) # identical perfect alignment, very different positional structure

    def test_density_invariant_images_triclinic(self):
        rng=np.random.default_rng(4)
        H=np.array([[1.,.2,.4],[0.,.8,.1],[0.,0.,1.2]])
        frac=rng.random((64,3));modes=np.array([[0,0,4],[1,-2,3],[2,1,0]])
        image=rng.integers(-4,5,size=(64,3))
        np.testing.assert_allclose(d.peak_order(frac,modes),d.peak_order(frac+image,modes),atol=1e-13)
        x=frac@H.T
        np.testing.assert_allclose(x@np.linalg.inv(H).T,frac,atol=1e-14)
        modes,_=d.density_modes(H,np.array([1.,2.,3.]),.04)
        self.assertTrue(len(modes)>0)
        self.assertTrue(np.all(np.mod(modes,1)==0))

    def test_triclinic_dump_and_partial_tail(self):
        text='''ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
-1 14 1
0 10 4
3 9 -2
ITEM: ATOMS id x y z vx vy vz c_shape[4] c_shape[5] c_shape[6] c_shape[7]
1 0 4 5 500 0 0 1 0 0 0
2 2 5 6 600 0 0 1 0 0 0
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
'''
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trajectory.lammpstrj';path.write_text(text)
            warnings=[];frames=list(d.read_frames(path,warnings))
        self.assertEqual(len(frames),1);self.assertEqual(len(warnings),1)
        np.testing.assert_allclose(frames[0].lattice,[[10,1,4],[0,8,-2],[0,0,6]])
        np.testing.assert_allclose(frames[0].origin,[-1,2,3])
        np.testing.assert_allclose(d.axes(frames[0]),[[0,0,1],[0,0,1]])

    def test_nonaffine_and_neighbor_image_velocity(self):
        H=np.array([[1.,0.,.4],[0.,1.,0.],[0.,0.,1.]])
        frac=np.array([[.2,.2,.95],[.2,.2,.05]])
        x=frac@H.T;v=np.zeros_like(x);v[:,0]=100*x[:,2]
        frame=d.Frame(0,np.zeros(3),H,['x','y','z','vx','vy','vz'],np.column_stack((x,v)))
        m=d.velocity_metrics(frame,x,np.array([0.,0.,1.]),100.,.2)
        self.assertAlmostEqual(m['velocity_nonaffine_rms_m_s'],0.)
        axis=np.tile([0.,0.,1.],(2,1))
        p=d.parallel_neighbor_motion(frame,x,axis,100.,.2,.05)
        self.assertEqual(p['parallel_neighbor_count'],1)
        self.assertAlmostEqual(p['parallel_neighbor_normal_center_speed_rms_m_s'],0.)
        self.assertAlmostEqual(p['parallel_neighbor_tangent_center_speed_rms_m_s'],10.,places=10)
        self.assertAlmostEqual(p['parallel_neighbor_affine_tangent_center_speed_rms_m_s'],10.,places=10)
        # Rewrap one particle, and remap its velocity consistently.
        x2=x.copy();v2=v.copy();x2[1]+=H[:,2];v2[1,0]+=100*H[2,2]
        frame2=d.Frame(0,np.zeros(3),H,frame.names,np.column_stack((x2,v2)))
        p2=d.parallel_neighbor_motion(frame2,x2,axis,100.,.2,.05)
        for k in p:
            self.assertAlmostEqual(p[k],p2[k],places=10)

    def test_quaternion_rotation_and_missing_columns(self):
        frame=d.Frame(0,np.zeros(3),np.eye(3),['quatw','quati','quatj','quatk'],
                      np.array([[np.sqrt(.5),0,np.sqrt(.5),0],[1,0,0,0]]))
        np.testing.assert_allclose(d.axes(frame),[[1,0,0],[0,0,1]],atol=1e-15)
        frame=d.Frame(0,np.zeros(3),np.eye(3),['x','y','z'],np.zeros((2,3)))
        self.assertIsNone(d.axes(frame))
        self.assertTrue(np.isnan(d.velocity_metrics(frame,np.zeros((2,3)),np.ones(3),100.,1.)['velocity_nonaffine_rms_m_s']))

if __name__=='__main__':
    unittest.main(verbosity=2)
