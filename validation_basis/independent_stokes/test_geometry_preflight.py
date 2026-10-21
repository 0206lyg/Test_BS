"""Actual spheroid polynomial checks; no Stokes solve or native backend needed."""
import unittest
from unittest import mock

import numpy as np

from geometry import build_mesh,spheroid_pose
import geometry_preflight as geometry


class GeometryPreflightTests(unittest.TestCase):
    def pose(self,ar=2,gap=1):
        return spheroid_pose(gap,aspect_ratio=ar,theta_i=0,theta_j=0,phi=0)

    def test_ar20_one_nm_selects_higher_polynomial_order_before_splitting(self):
        physical = dict(gap_nm=1.,aspect_ratio=20.,c_nm=150.,theta_i=0.,theta_j=0.,phi=0.)
        result = geometry.choose_orders(physical,dict(level=1,rim_levels=1,rim_width=.05,
            contact_levels=1,contact_radius=.25))
        self.assertEqual((result['coarse_order'],result['fine_order']),(10,12))
        self.assertFalse(result['requires_h_refinement'])
        self.assertTrue(all(x['max_position_error_over_gap']>.0005 for x in result['candidates'][:-1]))
        self.assertLess(result['candidates'][-1]['max_position_error_over_gap'],.0005)
        self.assertEqual(len(set(x['patch_count'] for x in result['candidates'])),1)

    def test_tensor_node_permutation_does_not_change_geometry_error(self):
        pose = self.pose()
        uv = geometry.gauss_uv(4)
        order = np.random.default_rng(20619).permutation(uv.shape[1])
        baseline = build_mesh(1.,4,0,uv,pose=pose)
        permuted = build_mesh(1.,4,0,np.asfortranarray(uv[:,order]),pose=pose)
        first = geometry.sample_geometry(baseline,uv,pose)
        second = geometry.sample_geometry(permuted,uv[:,order],pose)
        np.testing.assert_allclose(first['per_patch_position_error_over_gap'],
                                   second['per_patch_position_error_over_gap'],rtol=1e-12)

    def test_actual_selective_refinement_meets_tolerance(self):
        pose,uv = self.pose(),geometry.gauss_uv(6)
        def rebuild(descriptors=None):
            return build_mesh(1.,6,1,uv,pose=pose,patch_descriptors=descriptors)
        mesh,record = geometry.refine_to_tolerance(rebuild(),uv,pose,rebuild,max_refinements=2)
        self.assertEqual(record['status'],'passed')
        self.assertGreater(record['refinements'],0)
        self.assertLessEqual(record['attempts'][-1]['max_position_error_over_gap'],.0005)
        self.assertGreater(mesh['npts'],record['attempts'][0]['node_count'])
        self.assertFalse(record['physical_accuracy_qualified'])

    def test_memory_and_refinement_caps_do_not_launch_larger_mesh(self):
        pose,uv = self.pose(),geometry.gauss_uv(6)
        mesh = build_mesh(1.,6,1,uv,pose=pose)
        for budget,cap,expected in [(1.,3,'memory_budget_exceeded'),(56.,0,'refinement_limit')]:
            rebuild = mock.Mock()
            with self.assertRaises(geometry.GeometryPreflightError) as caught:
                geometry.refine_to_tolerance(mesh,uv,pose,rebuild,max_memory_gb=budget,
                                             max_refinements=cap)
            self.assertEqual(caught.exception.diagnostics['status'],expected)
            rebuild.assert_not_called()

    def test_wide_face_contact_sites_add_finite_boundary_samples(self):
        pose,uv = self.pose(20),geometry.gauss_uv(6)
        mesh = build_mesh(1.,6,1,uv,pose=pose)
        _,_,narrow_count = geometry.boundary_sites(mesh,pose,3,False)
        patches,checks,wide_count = geometry.boundary_sites(mesh,pose,3,True)
        self.assertGreater(wide_count,narrow_count)
        self.assertTrue(np.isfinite(checks).all())
        self.assertTrue(np.all(np.abs(checks)<1))
        self.assertEqual(checks.shape[1],len(patches))


if __name__=='__main__':
    unittest.main()
