"""Generic geometry/contracts only: no new AR Stokes solve or accuracy claim."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from numpy.polynomial.legendre import leggauss

import geometry as geo
import run_reference as driver


class GenericGeometry(unittest.TestCase):
    def test_recorded_defaults_preserve_existing_metadata_and_mesh(self):
        old = geo.recorded_pose(150.)
        new = geo.spheroid_pose(150.)
        self.assertIsInstance(old, geo.RecordedPose)
        self.assertTrue(new.is_recorded)
        self.assertEqual(old.metadata(), new.metadata())
        recorded = Path(__file__).parent/'results/pair_h150_p8_anisotropic_chunked.json'
        if recorded.exists():
            data = json.loads(recorded.read_text())
            for key, value in old.metadata().items():
                self.assertEqual(data[key], value, key)
        # Retain the original formula and its operation ordering, not only two
        # calls to the generalized code, as a default numerical regression.
        axes = np.array([[math.sin(geo.THETA_I), 0., math.cos(geo.THETA_I)],
            [math.sin(geo.THETA_J)*math.cos(geo.PHI), math.sin(geo.THETA_J)*math.sin(geo.PHI),
             math.cos(geo.THETA_J)]])
        axes /= np.linalg.norm(axes, axis=1)[:,None]
        tensors = np.stack([20.**2*np.eye(3)+(1-20.**2)*np.outer(q,q) for q in axes])
        centers = tensors[:,:,2]/np.sqrt(tensors[:,2,2])[:,None]*np.array([-1.,1.])[:,None]
        centers[1,2] += 1.
        np.testing.assert_array_equal(old.axes, axes)
        np.testing.assert_array_equal(old.centers, centers)
        x,_ = leggauss(4)
        u,v = np.meshgrid(x,x,indexing='ij'); uv = np.stack((u.ravel(),v.ravel()))
        a = geo.build_mesh(150.,3,0,uv)
        b = geo.build_mesh(150.,3,0,uv,pose=new)
        for key in ('srcvals','centers','axes','rotations','body_ids','norders','ixyzs','iptype'):
            np.testing.assert_array_equal(a[key],b[key])

    def test_generic_support_gap_volume_and_normals(self):
        for ar, ti, tj, az in ((5.,.3,1.1,2.4),(2.,math.pi/2,0.,0.),(1.,.5,2.,-1.)):
            pose = geo.spheroid_pose(21.,aspect_ratio=ar,c_nm=70.,theta_i=ti,theta_j=tj,phi=az)
            with self.subTest(ar=ar):
                self.assertFalse(pose.is_recorded)
                self.assertAlmostEqual(pose.gap_over_c,.3)
                np.testing.assert_allclose(np.einsum('bij,bkj->bik',pose.rotations,pose.rotations),
                                           np.broadcast_to(np.eye(3),(2,3,3)),atol=3e-15)
                for body, normal in ((0,[0,0,1]),(1,[0,0,-1])):
                    self.assertAlmostEqual(float(pose.implicit_residual(pose.contact_points[body],body)),0.,places=12)
                    np.testing.assert_allclose(pose.outward_normals(pose.contact_points[body],body),normal,atol=1e-13)
                nodes = geo.tensor_nodes(pose,96,64,phase=.31)
                for body in (0,1):
                    mask = nodes.body_ids == body
                    r = nodes.points[mask]-pose.centers[body]
                    volume = np.sum(nodes.weights[mask]*np.einsum('ij,ij->i',r,nodes.normals[mask]))/3.
                    self.assertLess(abs(volume/pose.exact_volume-1),3e-13)
                    self.assertLess(abs(nodes.weights[mask].sum()/pose.exact_area-1),1e-10)
                    np.testing.assert_allclose(nodes.normals[mask],pose.outward_normals(nodes.points[mask],body),atol=5e-14)
                self.assertAlmostEqual(pose.exact_volume,4*math.pi*ar**2/3)

    def test_full17_scaling_and_dimensional_similarity(self):
        pose = geo.spheroid_pose(12.,aspect_ratio=7.,c_nm=80.,theta_i=.2,theta_j=.8,phi=.7)
        scaled = geo.spheroid_pose(24.,aspect_ratio=7.,c_nm=160.,theta_i=.2,theta_j=.8,phi=.7)
        np.testing.assert_array_equal(pose.centers,scaled.centers)
        nodes = geo.tensor_nodes(pose,8,12)
        modes = geo.velocity_modes(nodes.points,nodes.body_ids,pose)
        np.testing.assert_array_equal(modes,geo.velocity_modes(nodes.points,nodes.body_ids,scaled))
        for body in (0,1):
            mask = nodes.body_ids == body; r = nodes.points[mask]-pose.centers[body]
            np.testing.assert_array_equal(modes[mask,:,6*body:6*body+3],np.broadcast_to(np.eye(3),(len(r),3,3)))
            for k in range(3):
                np.testing.assert_allclose(modes[mask,:,6*body+3+k],np.cross(np.eye(3)[k],r)/7.)
            for k,e in enumerate(geo.strain_basis()):
                np.testing.assert_allclose(modes[mask,:,12+k],-r@e.T/7.)
        # Quadrangular generation must propagate the pose rather than resetting AR20.
        quads = geo.quadrangular_nodes(pose,0,3)
        for body in (0,1):
            self.assertLess(np.abs(pose.implicit_residual(quads.points[quads.body_ids==body],body)).max(),1e-13)

    def test_invalid_physical_parameters_rejected(self):
        for kwargs in ({'aspect_ratio':.9},{'aspect_ratio':float('inf')},{'c_nm':0.},
                       {'c_nm':float('nan')},{'theta_i':-.01},{'theta_j':4.},
                       {'phi':float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                geo.spheroid_pose(1.,**kwargs)
        with self.assertRaises(ValueError): geo.spheroid_pose(-1.)
        x,_ = leggauss(3); u,v=np.meshgrid(x,x,indexing='ij'); uv=np.stack((u.ravel(),v.ravel()))
        with self.assertRaises(ValueError): geo.build_mesh(1.,2,0,uv,pose=geo.spheroid_pose(2.))
        with self.assertRaises(ValueError): geo.canonical_geometry(.1,ar=5.)

    def test_pose_identity_rejects_wrong_adaptation_dimensions_and_orientation(self):
        pose = geo.spheroid_pose(20.,aspect_ratio=5.,c_nm=100.,theta_i=.2,theta_j=.7,phi=1.)
        expected = pose.metadata()
        driver.validate_pose(expected,expected)
        for key in ('aspect_ratio','c_nm','theta_i','theta_j','phi','h_over_c'):
            changed = copy.deepcopy(expected); changed[key] += .1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,key):
                driver.validate_pose(changed,expected)
        # An actual --adapt-from dispatch must reject a different AR before mesh
        # assembly; the only backend interaction mocked is its UV-node request.
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary); parent=directory/'parent.json'
            changed=dict(expected,schema=driver.GENERIC_SCHEMA,convention_id=driver.CONVENTION)
            changed['aspect_ratio']=6.; parent.write_text(json.dumps(changed))
            backend=mock.Mock(); backend.discretization.return_value=(np.zeros((2,9)),)
            argv=['run_reference','--library',str(directory/'missing.so'),'--output',str(directory/'out.json'),
                  '--gap-nm','20','--aspect-ratio','5','--c-nm','100','--theta-i','.2','--theta-j','.7','--phi','1',
                  '--adapt-from',str(parent),'--order','2']
            with mock.patch.object(sys,'argv',argv), mock.patch.object(driver,'Backend',return_value=backend), \
                    mock.patch.object(driver,'build_mesh') as build, self.assertRaisesRegex(ValueError,'aspect_ratio'):
                driver.main()
            build.assert_not_called()

    def test_generic_isolated_evidence_keeps_same_pose_and_source_requirements(self):
        current=dict(geo.spheroid_pose(20.,aspect_ratio=5.,c_nm=100.,theta_i=.2,theta_j=.7,phi=1.).metadata(),
            schema=driver.GENERIC_SCHEMA,convention_id=driver.CONVENTION,library_sha256='fixture',alpha=1.,beta=1.,
            source_hashes={'geometry.py':'fixture_geometry','backend.py':'fixture_backend'})
        # Synthetic fixture is explicitly only a contract test, never output as
        # a physical reference file in results/.
        isolated=dict(current,status='complete',isolated_body=0,independent_reference=True,shares_production_solver=False,
            raw_full17=np.eye(17).tolist(),raw_reciprocity=0.,
            isolated_exact_validation={'passed':True,'block_relative_errors':dict(force=0.,torque=0.,stresslet=0.)},
            validation=dict(off_grid_boundary=True,boundary_evaluation_valid=True,boundary_max_scaled_error=0.,
                            exact_surface_position_max_error_over_gap=0.,linear_relative_residual=0.))
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'isolated.json'
            for defect in (None,'aspect_ratio','c_nm','schema','source'):
                data=copy.deepcopy(isolated)
                if defect in ('aspect_ratio','c_nm'): data[defect] += 1
                elif defect=='schema': data['schema']=driver.SCHEMA
                elif defect=='source': data['source_hashes']['geometry.py']='wrong'
                path.write_text(json.dumps(data))
                with self.subTest(defect=defect):
                    self.assertEqual(driver.isolated_evidence(path,current)['passed'],defect is None)


if __name__ == '__main__':
    unittest.main()
