"""Geometry-only sampling tests; no BIE or physical accuracy claim."""
import json
import math
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from contact_check_sites import build_contact_check_sites
from geometry import build_mesh, spheroid_pose


def mesh_for(pose, isolated_body=None):
    x, _ = leggauss(5)
    u, v = np.meshgrid(x, x, indexing='ij')
    return build_mesh(pose.gap_nm, 4, 1, np.stack((u.ravel(), v.ravel())),
                      rim_levels=1, contact_levels=1, pose=pose,
                      isolated_body=isolated_body)


def legacy_sites(mesh, pose):
    ids, uv = [], []
    for item in mesh['contact_patch_id_uv']:
        direction = np.asarray(item['direction'])
        tangent = np.cross(direction, [1., 0., 0.])
        tangent /= np.linalg.norm(tangent)
        other = np.cross(direction, tangent)
        directions = [direction]
        width = np.sqrt(pose.gap_over_c / pose.a)
        for radius in [.25 * width, width]:
            for angle in np.arange(8) * np.pi / 4 + .173:
                d = direction + radius * (np.cos(angle) * tangent + np.sin(angle) * other)
                directions.append(d / np.linalg.norm(d))
        for d in directions:
            pid, target = mesh['locate_direction'](item['body'], d)
            ids.append(pid)
            uv.append(np.clip(target, -1 + 1e-3, 1 - 1e-3))
    return np.asarray(ids), np.asarray(uv).T


class ContactCheckSites(unittest.TestCase):
    def test_legacy_targets_preserved_before_additions(self):
        for pose in (spheroid_pose(150.),
                     spheroid_pose(1., aspect_ratio=2, theta_i=0, theta_j=0, phi=0),
                     spheroid_pose(4., aspect_ratio=5, theta_i=.7, theta_j=1.1, phi=2.)):
            mesh = mesh_for(pose)
            expected_ids, expected_uv = legacy_sites(mesh, pose)
            ids, uv, meta = build_contact_check_sites(mesh, pose)
            count = len(expected_ids)
            np.testing.assert_array_equal(ids[:count], expected_ids)
            np.testing.assert_array_equal(uv[:, :count], expected_uv)
            self.assertEqual(count, 34)
            self.assertEqual(meta['legacy_site_count'], count)
            self.assertEqual(meta['hessian_site_count'], 32)
            self.assertEqual(ids.dtype, np.int64)
            self.assertTrue(uv.flags.f_contiguous)

    def test_face_and_rim_widths_follow_exact_known_curvatures(self):
        for ar in (2., 5., 10., 20.):
            for face in (True, False):
                theta = 0. if face else math.pi / 2
                pose = spheroid_pose(1., aspect_ratio=ar, theta_i=theta, theta_j=theta, phi=0)
                ids, uv, meta = build_contact_check_sites(mesh_for(pose), pose)
                radii = np.sort(meta['gap_doubling_ellipse_semiaxes_c'])
                expected = ([ar * math.sqrt(pose.gap_over_c)] * 2 if face else
                            [math.sqrt(pose.gap_over_c / ar), math.sqrt(pose.gap_over_c * ar)])
                np.testing.assert_allclose(radii, expected, rtol=1e-12)
                self.assertEqual(meta['projection_clipped_ray_count'], 0)

    def test_generic_targets_stay_on_requested_bodies_and_inside_patches(self):
        for ar in (2., 5., 10., 20.):
            for angles in ((0., 0., 0.), (0., math.pi / 2, 0.),
                           (math.pi / 2, math.pi / 2, math.pi / 2), (.35, .8, 1.2)):
                for gap in (1., 4., 150.):
                    pose = spheroid_pose(gap, aspect_ratio=ar, theta_i=angles[0],
                                         theta_j=angles[1], phi=angles[2])
                    mesh = mesh_for(pose)
                    ids, uv, meta = build_contact_check_sites(mesh, pose)
                    self.assertTrue(np.isfinite(uv).all())
                    self.assertLessEqual(float(np.abs(uv).max()), 1 - 1e-3)
                    self.assertEqual(uv.shape, (2, len(ids)))
                    for k, pid in enumerate(ids):
                        body = int(mesh['patches'][pid][0])
                        point = mesh['exact_patch_values'](int(pid), uv[:, k:k + 1])[:3, 0]
                        self.assertLess(abs(float(pose.implicit_residual(point, body))), 1e-10)
                    json.dumps(meta, allow_nan=False)

    def test_far_gap_clipping_and_isolated_body(self):
        pose = spheroid_pose(1e7, aspect_ratio=20, theta_i=.35, theta_j=.8, phi=1.2)
        mesh = mesh_for(pose, isolated_body=1)
        ids, uv, meta = build_contact_check_sites(mesh, pose)
        self.assertEqual(meta['body_ids'], [1])
        self.assertEqual(meta['legacy_site_count'], 17)
        self.assertEqual(meta['hessian_site_count'], 16)
        self.assertEqual(meta['projection_clipped_ray_count'], 16)
        self.assertLess(meta['maximum_used_xy_radius_c'], 40.)
        self.assertTrue(all(mesh['patches'][pid][0] == 1 for pid in ids))

    def test_invalid_controls_rejected(self):
        pose = spheroid_pose(1.)
        mesh = mesh_for(pose)
        for controls in ({'edge_avoidance': 0}, {'projection_fraction': 1},
                         {'ring_factors': (float('nan'),)}, {'ring_factors': ()},
                         {'azimuth_count': 2}, {'azimuth_count': True}):
            with self.assertRaises(ValueError):
                build_contact_check_sites(mesh, pose, **controls)


if __name__ == '__main__':
    unittest.main()
