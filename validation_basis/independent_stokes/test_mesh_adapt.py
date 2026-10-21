"""Small geometry/indicator tests; no Stokes solve or physical pass."""
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss,legval

from geometry import THETA_I,THETA_J,PHI,cube_sphere_patches
from mesh_adapt import density_tails,propose


class AdaptationTests(unittest.TestCase):
    def test_directional_indicator_with_permuted_backend_nodes(self):
        x,_ = leggauss(9)
        u,v = np.meshgrid(x,x,indexing="ij")
        uv = np.stack((u.ravel(),v.ravel()))
        permutation = np.random.default_rng(20).permutation(81)
        uv = uv[:,permutation]
        coefficients = np.zeros(9);coefficients[-1] = 1
        density = np.zeros((162,3,17))
        density[:81,0,0] = legval(uv[0],coefficients)
        density[81:,0,0] = legval(uv[1],coefficients)
        tails = density_tails(density,uv,2)
        np.testing.assert_allclose(tails,[[1,0],[0,1]],atol=1e-13)

    def test_proposal_preserves_exact_cube_face_cover_and_budget(self):
        cube = cube_sphere_patches(1)
        descriptors = [[b,vertices.tolist()] for b in (0,1) for vertices in cube]
        record = {"aspect_ratio":20.0,"theta_i":THETA_I,"theta_j":THETA_J,
                  "phi":PHI,"h_over_c":1.0,"h_nm":150.0,"focus":0.0,
                  "isolated_body":None,"order":8,"patch_descriptors":descriptors,
                  "validation":{"boundary_per_patch_max_scaled_error":np.linspace(.001,.1,48).tolist()}}
        tails = np.tile([.001,.1],(48,1));tails[::3] = [.1,.1]
        result = propose(record,tails,max_nodes=8000)
        self.assertLessEqual(result["node_count"],8000)
        self.assertFalse(result["physical_accuracy_qualified"])
        def face_areas(patches):
            area = np.zeros((2,3,2))
            for body,vertices in patches:
                q = np.array(vertices)
                axis = int(np.argmin(np.ptp(q,axis=0)))
                sign = int(q[0,axis]>0)
                area[body,axis,sign] += np.linalg.norm(np.cross(q[1]-q[0],q[3]-q[0]))
            return area
        np.testing.assert_array_equal(face_areas(result["patch_descriptors"]),face_areas(descriptors))


if __name__ == "__main__":
    unittest.main()
