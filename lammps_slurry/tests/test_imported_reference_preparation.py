"""Ephemeral manufactured artifact contracts; no Stokes or rheology validation."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('prepare_imported', Path(__file__).parents[1]/'tools/prepare_case.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
RAW_MODEL = 'JANOSCHEK_MATCHED_HARDFLOOR_FULL17_V4'
NEW_MODEL = 'JANOSCHEK_DISSIPATIVE_INCREMENT_FULL17_V1'
CONVENTION = 'slurry_full17_c_eta_1_v1'


class ImportedPreparation(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='test_only_imported_pipeline_')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.args = SimpleNamespace(ar=20., lubrication_epsilon_nm=1., finite_gap_range=.05,
            finite_gap_fast_nphi=4, finite_gap_fast_reference_pipeline=self.root/'pipeline.json',
            finite_gap_fast_table=None, finite_gap_fast_source=None, finite_gap_fast_generator=None)
        self.a = 3e-6
        theta = [0., math.pi/6, math.pi/3, math.pi/2]
        phi = [-math.pi, -math.pi/2, 0., math.pi/2]
        lower = 1/150
        header = ['SLURRY_SPD_LOGFOURIER_V4', 'MODEL '+NEW_MODEL, f'20 {lower} 1 4 4 4', '0',
                  ' '.join(map(str,theta)), ' '.join(map(str,phi)), f'{lower} .02 .2 1', 'DATA']
        (self.root/'reference.sft').write_bytes(('\n'.join(header)+'\n').encode()+b'\0'*(4**4*153*8))
        (self.root/'reference.source17').write_text('TEST_ONLY_MANUFACTURED_SOURCE_IDENTITY\n')
        original = dict(independent_reference=True,shares_production_solver=False,test_fixture_only=True)
        rows = [dict(provisional_accepted=True,physical_accuracy_qualified=False,
                     original_metadata=copy.deepcopy(original)) for _ in range(64)]
        self.source = dict(schema='slurry_reference_source17_manifest_v1', source_format='SLURRY_REFERENCE_SOURCE17_V1',
            model=RAW_MODEL, convention_id=CONVENTION, status='PROVISIONAL',
            physical_accuracy_qualified=False,strict_physical_qualification=False,
            source_text_sha256=prepare.file_sha256(self.root/'reference.source17'),
            layout=dict(layout='full-grid',complete_grid_verified=True),
            aspect_ratio=20.,c_m=150e-9,floor_m=1e-9,anchor_h_over_c=1.,pose_count=64,sources=rows,
            development_boundary_acceptance_limit=.01,test_fixture_only=True)
        self.generation = dict(schema='reference_source17_table_generation_v1', status='provisional_numerical_transport_passed',
            physical_accuracy_qualified=False,continuation_physically_validated=False,
            contains_new_stokes_solve=False,angular_interpolation_supported=True,angular_interpolation_physically_validated=False,
            matched_model='dissipative-increment',model_revision=NEW_MODEL,convention=CONVENTION,
            layout='complete_actual_source_grid',aspect_ratio=20.,c_m=150e-9,floor_m=1e-9,hmin_over_c=lower,hmax_over_c=1.,
            pose_count=64,gap_nodes=4,matrix_dimension=17,stored_values_per_node=153,
            maximum_gap_interpolation_relative_energy_error=1e-8,interpolation_tolerance=1e-6,
            source_fnv1a64=prepare.reference_fnv1a64(self.root/'reference.source17'),test_fixture_only=True,
            numerical_source_fingerprint='fixture_preparation_source')
        self.runtime = dict(schema='reference17_actual_pipeline_v1',runtime_layout='complete_actual_source_grid',
            physical_accuracy_qualified=False,pipeline_algebra_passed=True,poses_checked=64,
            source_binding='external_pipeline_manifest_required',
            source_fnv1a64=self.generation['source_fnv1a64'],table_fnv1a64=prepare.reference_fnv1a64(self.root/'reference.sft'),
            model_revision=NEW_MODEL,test_fixture_only=True,numerical_source_fingerprint='fixture_preparation_source')
        self.pipeline = dict(schema='provisional_reference_table_pipeline_v1',status='complete',workflow_completed=True,
            layout='full-grid',matched_model='dissipative-increment',physical_accuracy_qualified=False,
            stages=[dict(stage=name,returncode=0) for name in ('import','generate','runtime')],test_fixture_only=True)
        self.save()

    def save(self):
        for name, data in [('reference.source17.manifest.json',self.source),('generation.json',self.generation),('runtime.json',self.runtime)]:
            (self.root/name).write_text(json.dumps(data))
        self.pipeline['outputs'] = {p.name:dict(bytes=p.stat().st_size,sha256=prepare.file_sha256(p))
                                    for p in self.root.iterdir() if p.name!='pipeline.json'}
        (self.root/'pipeline.json').write_text(json.dumps(self.pipeline))

    def resolve(self):
        # This must not trigger old basis creation or any particle/MD work.
        with mock.patch.object(prepare,'resolve_fast_basis',side_effect=AssertionError('legacy basis must not run')), \
             mock.patch.object(prepare,'fast_source_fingerprint',return_value='fixture_preparation_source'):
            return prepare.resolve_fast_table(self.args,self.a)

    def test_explicit_complete_grid_preserves_provisional_status_and_actual_revision(self):
        self.assertEqual(self.resolve(),self.root/'reference.sft')
        manifest=self.args.finite_gap_fast_manifest
        self.assertFalse(manifest['physical_accuracy_qualified'])
        self.assertEqual(manifest['source_gate']['status'],'not_qualified')
        self.assertEqual(manifest['source_numerical_gate']['status'],'provisional_accepted')
        self.assertEqual(manifest['interpolation_gate']['status'],'passed')
        self.assertFalse(manifest['interpolation_gate']['angular_accuracy_qualified'])
        self.assertEqual(manifest['table']['header']['model_revision'],NEW_MODEL)
        self.assertEqual(manifest['table']['sha256'],prepare.file_sha256(self.root/'reference.sft'))

    def test_new_model_header_requires_explicit_imported_route(self):
        with self.assertRaisesRegex(ValueError,'revision mismatch'):
            prepare.read_fast_table_header(self.root/'reference.sft',self.args,self.a)
        self.assertEqual(prepare.read_fast_table_header(self.root/'reference.sft',self.args,self.a,
            imported_model_revision=NEW_MODEL)['model_revision'],NEW_MODEL)

    def test_sparse_failed_or_incomplete_pipeline_rejected_before_basis(self):
        for field, value in [('layout','sparse'),('workflow_completed',False),('status','failed'),
                             ('physical_accuracy_qualified',True)]:
            old=self.pipeline[field];self.pipeline[field]=value;self.save()
            with self.subTest(field=field),self.assertRaises(ValueError): self.resolve()
            self.pipeline[field]=old
        self.pipeline['stages'][2]['returncode']=2;self.save()
        with self.assertRaisesRegex(ValueError,'stages'): self.resolve()

    def test_all_named_outputs_are_hash_checked(self):
        (self.root/'extra.stdout.log').write_text('bound diagnostic')
        self.save()
        (self.root/'extra.stdout.log').write_text('altered diagnostic')
        with self.assertRaisesRegex(ValueError,'size/SHA256'): self.resolve()

    def test_physical_size_floor_ar_range_and_nphi_mismatches_rejected(self):
        for key,value in [('ar',19.),('lubrication_epsilon_nm',2.),('finite_gap_range',.051),('finite_gap_fast_nphi',8)]:
            old=getattr(self.args,key);setattr(self.args,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError): self.resolve()
            setattr(self.args,key,old)
        with self.assertRaisesRegex(ValueError,'particle size'):
            with mock.patch.object(prepare,'fast_source_fingerprint',return_value='fixture_preparation_source'):
                prepare.resolve_fast_table(self.args,2.99e-6)

    def test_model_runtime_identity_and_synthetic_source_rejected(self):
        cases=[(self.runtime,'pipeline_algebra_passed',False),(self.runtime,'poses_checked',63),
               (self.runtime,'runtime_layout','sparse_exact_pose'),(self.runtime,'source_fnv1a64','0'*16),
               (self.runtime,'table_fnv1a64','0'*16),(self.generation,'model_revision',RAW_MODEL),
               (self.source,'strict_physical_qualification',True)]
        for record,key,value in cases:
            old=record[key];record[key]=value;self.save()
            with self.subTest(key=key),self.assertRaises(ValueError):self.resolve()
            record[key]=old
        self.source['sources'][0]['original_metadata']['synthetic']=True;self.save()
        with self.assertRaisesRegex(ValueError,'actual independent'):self.resolve()

    def test_stale_generation_and_runtime_fingerprints_rejected(self):
        for record in (self.generation,self.runtime):
            record['numerical_source_fingerprint']='stale';self.save()
            with self.assertRaisesRegex(ValueError,'fingerprint is stale'):
                self.resolve()
            record['numerical_source_fingerprint']='fixture_preparation_source'

    def test_copied_artifact_is_checked_against_manifest_and_copy_header(self):
        source=self.resolve(); destination=self.root/'run-table.sft'
        # Removing the original after copy must not matter: validation reads
        # the exact file that will be loaded by the run.
        actual_copy=prepare.shutil.copyfile
        def copy_then_remove(src,dst):
            result=actual_copy(src,dst)
            Path(src).unlink()
            return result
        with mock.patch.object(prepare.shutil,'copyfile',side_effect=copy_then_remove):
            sha,header=prepare.copy_fast_table_checked(source,destination,self.args,self.a)
        self.assertEqual(sha,self.args.finite_gap_fast_manifest['table']['sha256'])
        self.assertEqual(header['model_revision'],NEW_MODEL)
        source.write_bytes(destination.read_bytes()+b'changed after validation')
        with self.assertRaisesRegex(ValueError,'copied table SHA256'):
            prepare.copy_fast_table_checked(source,self.root/'changed-run-table.sft',self.args,self.a)

    def test_source_grid_and_existing_table_conflicts_rejected(self):
        self.source['layout']['complete_grid_verified']=False;self.save()
        with self.assertRaisesRegex(ValueError,'complete actual angular grid'):self.resolve()
        self.source['layout']['complete_grid_verified']=True;self.save()
        self.args.finite_gap_fast_source=self.root/'old.ssb'
        with self.assertRaisesRegex(ValueError,'legacy source'):self.resolve()
        self.args.finite_gap_fast_source=None
        self.args.finite_gap_fast_table=self.root/'wrong.sft'
        with self.assertRaisesRegex(ValueError,'differs'):self.resolve()


if __name__=='__main__':
    unittest.main()
