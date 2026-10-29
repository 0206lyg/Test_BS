"""Synthetic algebra and evidence-contract tests, not physical validation."""
import json
import csv
import math
import unittest
import numpy as np
import distance_regime_assessment as assess
from test_assess_campaign_accuracy import CampaignAccuracyTests

class DistanceRegimeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = CampaignAccuracyTests(); self.fixture.setUp()
        self.fixture.make_reference('coarse', np.eye(17)*1.0001, boundary=.00001)
        self.fixture.make_reference('fine', np.eye(17), boundary=.00001)
        self.point = dict(id='AR5_tilted_h30', aspect_ratio=5., c_nm=150., gap_nm=30.,
                          theta_i=.2,theta_j=.3,phi=.4,orientation='tilted')
        self.operators = dict(convention_id=assess.CONVENTION, independent_reference=False,
            aspect_ratio=5., c_nm=150., h_over_c=.2, theta_i=.2,theta_j=.3,phi=.4,
            near_total17=(1.005*np.eye(17)).tolist(),far_isolated17=(1.5*np.eye(17)).tolist(),
            leading17=(.1*np.eye(17)).tolist())
    def tearDown(self): self.fixture.tearDown()
    def run_point(self):
        return assess.assess_point(self.point,[self.fixture.coarse,self.fixture.fine],self.operators)
    def test_sample_classification_requires_validated_reference(self):
        result=self.run_point()
        self.assertTrue(result['reference_resolved'],result['reasons'])
        self.assertEqual(result['status'],'NEAR'); self.assertFalse(result['physical_accuracy_qualified'])
        self.operators['far_isolated17']=np.eye(17).tolist()
        self.assertEqual(self.run_point()['status'],'FAR')
        self.operators['near_total17']=(2*np.eye(17)).tolist()
        self.assertEqual(self.run_point()['status'],'FAR')
        self.operators['far_isolated17']=(2*np.eye(17)).tolist()
        self.assertEqual(self.run_point()['status'],'INTERMEDIATE')
    def test_bad_isolated_evidence_is_unresolved_not_model_failure(self):
        data=json.loads(self.fixture.fine.read_text())
        data['moments']['isolated_spheroid_validation_passed']=False
        self.fixture.write(self.fixture.fine,data); result=self.run_point()
        self.assertEqual(result['status'],'UNRESOLVED'); self.assertFalse(result['reference_resolved'])
        self.assertTrue(any('isolated_moment' in reason for reason in result['reasons']))
        self.assertIn('metrics',result['models']['near'])
    def test_geometry_change_and_reused_mesh_cannot_pass(self):
        self.operators['h_over_c']=1.; result=self.run_point()
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertTrue(any('geometry' in reason for reason in result['reasons']))
        self.operators['h_over_c']=.2
        result=assess.assess_point(self.point,[self.fixture.fine,self.fixture.fine],self.operators)
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertIn('coarse_fine_report_reused',result['reasons'])
    def test_weak_antisymmetric_mode_not_hidden_by_squeeze(self):
        reference=np.diag(np.geomspace(1,1e12,17)); predicted=reference.copy()
        scale=math.sqrt(reference[0,0]*reference[1,1])
        predicted[0,1]=.1*scale; predicted[1,0]=-.1*scale
        metrics=assess.balanced_metrics(predicted,reference)
        self.assertAlmostEqual(metrics['full17_relative_energy_error'],0.)
        self.assertAlmostEqual(metrics['energy_normalized_full_response_operator_error'],.1)
        self.assertLess(metrics['raw_response_relative_frobenius_error'],1e-10)
        self.assertEqual(assess.model_comparison(predicted,reference,.01,'test')['status'],'FAIL')
    def test_indefinite_near_model_is_not_repaired(self):
        near=np.eye(17); near[0,0]=-1; self.operators['near_total17']=near.tolist()
        result=self.run_point(); self.assertEqual(result['status'],'INTERMEDIATE')
        self.assertFalse(result['models']['near']['positive_dissipation'])
        self.assertLess(result['models']['near']['metrics']['predicted_minimum_symmetric_eigenvalue'],0)
    def test_adapted_p_refinement_can_have_fewer_nodes(self):
        data=json.loads(self.fixture.coarse.read_text()); data['node_count']=400
        self.fixture.write(self.fixture.coarse,data); result=self.run_point()
        self.assertTrue(result['reference_resolved'],result['reasons'])
        self.assertFalse(result['spatial_refinement_design']['total_node_count_increased'])
        self.assertTrue(result['spatial_refinement_design']['polynomial_order_increased'])
    def test_refinement_failure_and_missing_report_remain_unresolved(self):
        self.fixture.make_reference('coarse',1.1*np.eye(17),boundary=.00001)
        result=self.run_point(); self.assertEqual(result['status'],'UNRESOLVED')
        self.assertIn('paired_spatial_resolution_change_exceeds_tolerance',result['reasons'])
        self.fixture.fine.unlink(); result=self.run_point()
        self.assertEqual(result['status'],'UNRESOLVED'); self.assertFalse(result['comparison_uses_requested_finest'])
    def anchor_fixture(self):
        rows=[]; raws={}; operators={}
        for gap in (1.,3.,10.,30.):
            rows.append(dict(id=str(gap),gap_nm=gap,aspect_ratio=5.,orientation='tilted',
                geometry=dict(aspect_ratio=5.,c_m=150e-9,theta_i=.2,theta_j=.3,phi=.4,h_over_c=gap/150),
                reference_resolved=True,status='INTERMEDIATE',models={
                    'near':dict(status='FAIL',passed=False), 'far':dict(status='FAIL',passed=False)}))
            operators[str(gap)]=dict(leading17=(np.eye(17)/gap).tolist())
            raws[str(gap)]=np.eye(17)/gap+np.eye(17)*2
        return rows,raws,operators
    def test_anchor_never_validates_itself_and_needs_two_lower_samples(self):
        rows,raws,operators=self.anchor_fixture()
        trials=assess.matched_anchor_trials(rows,raws,operators,selected_gap_nm=10.)
        trial=next(t for t in trials if t['anchor_gap_nm']==10.)
        self.assertEqual([v['gap_nm'] for v in trial['heldout']],[1.,3.])
        self.assertTrue(trial['sampled_overlap_supported'])
        self.assertEqual([p['status'] for p in rows],['NEAR','NEAR','INTERMEDIATE','INTERMEDIATE'])
        trial=next(t for t in trials if t['anchor_gap_nm']==3.)
        self.assertFalse(trial['sampled_overlap_supported']); self.assertFalse(trial['physical_accuracy_qualified'])
    def test_unresolved_gap_breaks_contiguous_anchor_support(self):
        rows,raws,operators=self.anchor_fixture(); rows[1]['reference_resolved']=False
        trials=assess.matched_anchor_trials(rows,raws,operators,selected_gap_nm=30.)
        trial=next(t for t in trials if t['anchor_gap_nm']==30.)
        self.assertFalse(trial['sampled_overlap_supported'])
        self.assertEqual(trial['consecutive_passing_gap_groups_nm'],[[1.],[10.]])
        self.assertEqual(rows[0]['status'],'UNRESOLVED')
    def test_overlap_means_near_and_stokes_across_distances(self):
        rows,raws,operators=self.anchor_fixture()
        for row in rows[:2]: row['models']['near']=dict(status='PASS',passed=True)
        summary=assess.assess_points(rows)
        self.assertEqual([row['status'] for row in rows],['OVERLAP','OVERLAP','INTERMEDIATE','INTERMEDIATE'])
        self.assertEqual(summary['aspects'][0]['orientations']['tilted']['near_stokes_overlap_sample_groups_nm'],[[1.,3.]])
        rows[1]['reference_resolved']=False; assess.assess_points(rows)
        self.assertEqual(rows[0]['status'],'NEAR')
    def test_far_cutoff_requires_multiple_outer_samples_and_all_poses(self):
        rows,_,_=self.anchor_fixture()
        for row in rows:
            row['status']='FAR'; row['models']['far']=dict(status='PASS',passed=True)
        result=assess.assess_points(rows,['tilted','face_face'],[1.,3.,10.,30.])
        pose=result['aspects'][0]['orientations']['tilted']
        self.assertEqual(pose['far_onset_candidate_nm'],1.)
        self.assertFalse(pose['far_onset_is_continuous_cutoff'])
        self.assertTrue(all(not r['all_far_pass'] for r in result['aspects'][0]['common_sample_coverage']))
        rows[-1]['status']='UNRESOLVED'; rows[-1]['reference_resolved']=False; result=assess.assess_points(rows)
        self.assertIsNone(result['aspects'][0]['orientations']['tilted']['far_onset_candidate_nm'])
    def test_far_pass_survives_optional_unresolved_selected_anchor(self):
        rows,_,_=self.anchor_fixture()
        rows[0]['models']['far']=dict(status='PASS',passed=True)
        rows[0]['models']['selected_matched_near']=dict(status='UNRESOLVED',passed=False)
        self.assertEqual(assess.classify(rows[0]),'FAR')
    def test_missing_campaign_evidence_still_writes_readable_status(self):
        points=[dict(self.point,reference_task_ids=[0,1])]
        tasks=[dict(task_id=k,output_subdir=f'tasks/{k}') for k in (0,1)]
        campaign=self.fixture.folder/'campaign.json'
        self.fixture.write(campaign,dict(points=points,tasks=tasks,settings={}))
        output=self.fixture.folder/'regimes.json'
        result=assess.collect_and_assess(campaign,self.fixture.folder/'not_needed_exporter',output)
        self.assertEqual(result['counts']['UNRESOLVED'],1)
        self.assertTrue(output.with_suffix('.md').is_file())
        self.assertIn('0/1',output.with_suffix('.md').read_text())
        with output.with_suffix('.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(rows[0]['status'],'UNRESOLVED')
        self.assertIn('near_stresslet_error',rows[0])

if __name__=='__main__': unittest.main()
