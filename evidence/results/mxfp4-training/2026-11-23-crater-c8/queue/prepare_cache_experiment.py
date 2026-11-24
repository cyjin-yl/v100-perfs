#!/usr/bin/env python3
"""One-shot source-bound C2 preparation cache OFF/ON experiment.

All builds use one frozen source archive. CUDA execution waits for the completed
RNE-v2 archive and the exact same owned B flock. Native full suites, the actual
preparation microcheck, Torch certified16 and CUDA RNE135 on both builds precede performance.
No remote calls, retries, packages, clock changes, or unrelated process cleanup.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import sys
import tarfile
import time
import types

HELPER = Path('/workspace/user/mxfp4-c9-c10-20261123T2128.py')
HELPER_SHA = 'f7f45628bcf7616a7e7915823c77f2f75602cff66be29bf640c5ec3be8ecce1e'
QUANTIZER_HELPER_SHA = '96d3db40d25426370cfc8542dff908edf00a385c0055019c727c9c005fb8a548'
DEFAULT_PRIOR = '/workspace/user/mxfp4-cuda-rne-v2-20261123T2204'
SOURCE_COMMIT = 'c581389551be2233ba23404fa72c471cc4c6f961'
SOURCE_SHA = 'a7062abcfa1b9263d7a7bf0d282f2de1b51cfa279939f3b969eacb543f5417c0'
SOURCE_BYTES = 146431
SOURCE_FILES = 30
MICRO_SHA = 'b2c4370afef07d5f4ef0539c0b4aea0f200098dafabc801c3885dd61caee6d42'
TAGS = ('off', 'on')
SHAPES = (2048, 4096, 8192)
MODES = {2048: 'random_block_scales', 4096: 'random_block_scales', 8192: 'random_unit_scale'}
ORDER = ('off', 'on', 'on', 'off')
H = Q = None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): result.update(block)
    return result.hexdigest()


def reviewed_module(path, expected, name):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise RuntimeError('Reviewed helper is not an owned nonsymlink regular file: ' + str(path))
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise RuntimeError('Reviewed helper SHA differs: ' + str(path))
    module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def planned_names():
    return ['build-off', 'build-on', 'build-prepare-micro', 'wait-rnev2-lock'] + [
        f'check-{tag}-{kind}' for tag in TAGS for kind in ('baseline', 'tiled', 'certified')] + [
        'check-prepare-micro', 'torch-certified-on', 'quantize-cuda-off', 'quantize-cuda-on'] + [
        f'bench-{size}-{index+1:02d}-{tag}' for size in SHAPES for index, tag in enumerate(ORDER)]


def micro_gate(rows):
    environment = H.one(rows, 'prepare_cache_environment')
    H.fields(environment, {'actual_V100': True, 'certified_variant': 2, 'certified_prepare_raw_cache': True,
        'explicit_allocation_cap_bytes': 8*1024**2, 'reserve_bytes': 512*1024**2,
        'minimum_free_bytes': 1024**3, 'shared_dispatch_helper_checks': 7,
        'performance_timing_executed': False}, 'Prepare micro environment')
    summary = H.one(rows, 'prepare_cache_summary')
    H.fields(summary, {'status': 'passed', 'passed': True, 'gpu_executed': True,
        'all_submitted_work_completed': True, 'actual_V100': True, 'certified_variant': 2,
        'certified_prepare_raw_cache': True, 'cases': 327, 'direct_cached_vs_original_cases': 323,
        'public_dispatch_cases': 4, 'public_fallback_preparation_cases': 2,
        'public_cached_preparation_eligible_cases': 2, 'normalized_half_values': 1079840,
        'rowdata_rows': 1635, 'public_FP32_output_values': 24, 'normalized_half_bit_mismatches': 0,
        'rowdata_byte_mismatches': 0, 'independent_cpu_oracle_mismatches': 0,
        'public_output_bit_mismatches': 0, 'guard_bytes_unchanged': True, 'input_bytes_unchanged': True,
        'kernel_path_instrumented': False, 'performance_timing_executed': False}, 'Prepare micro summary')
    H.need(0 < summary.get('peak_explicit_allocation_bytes', 0) <= 8*1024**2, 'Prepare micro allocation bound differs')
    cases = [r for r in rows if r.get('type') == 'prepare_cache_correctness']
    H.need(len(cases) == len({r.get('name') for r in cases}) == 327, 'Prepare micro case count/name duplication')
    H.need(sum(r.get('normalized_half_values', 0) for r in cases) == 1079840 and
           sum(r.get('rowdata_rows', 0) for r in cases) == 1635, 'Prepare micro actual totals differ')
    for case in cases:
        H.fields(case, {'passed': True, 'gpu_executed': True, 'M': 3, 'N': 2,
            'rowdata_rows': 5, 'rowdata_bytes_per_row': 16, 'normalized_half_bit_mismatches': 0,
            'rowdata_byte_mismatches': 0, 'independent_cpu_oracle_mismatches': 0,
            'input_bytes_unchanged': True, 'guard_bytes_unchanged': True,
            'actual_original_kernel_compared': True, 'kernel_path_instrumented': False,
            'poison': 'bitwise_inverse_of_every_CPU_oracle_output_byte'}, 'Prepare micro actual case')
        H.need(case['normalized_half_values'] == 5*case['K'], 'Prepare micro half count differs')
    public = [r for r in cases if r.get('public_dispatch_executed') is True]
    direct = [r for r in cases if r.get('actual_cache_kernel_directly_invoked') is True]
    H.need(len(public) == 4 and {r['K'] for r in public} == {0, 32, 8192, 8224}
           and len(direct) == 323 and all(r.get('shared_public_dispatch_helper_eligible') is True for r in direct),
           'Actual cache/public dispatch case set differs')
    for case in public:
        H.need(case['shared_public_dispatch_helper_eligible'] is (case['K'] in (32,8192)),
               'Public fallback was misreported as cache-eligible')
    return summary


def create_runner_class():
    ReviewedQuantizerRunner = Q.create_runner_class()
    class CacheRunner(ReviewedQuantizerRunner):
        def __init__(self, args, run):
            # Explicitly bypass the RNE-v2 constructor's old source/20-stage plan.
            H.Runner.__init__(self, args, run)
            self.own_hash = H.sha(Path(__file__)); self.binding_hash = None
            self.qhelper = Path(args.prior_quantizer_run) / 'results/quantizer-experiment-runner.py'
            self.library = self.run / 'build-on/libmxfp4_sm70.so'
            self.manifest = self.run / 'build-on/compile-evidence.json'
            self.report = {'schema_version': 1, 'purpose': 'C2 raw preparation cache OFF/ON complete pipeline experiment',
                'status': 'running', 'passed': False, 'started_at_utc': H.utc(), 'run_directory': str(run),
                'python_execution_path': args.python, 'nvcc': H.NVCC, 'runner_sha256': self.own_hash,
                'reviewed_native_helper_path': str(HELPER), 'reviewed_native_helper_sha256': HELPER_SHA,
                'reviewed_quantizer_helper_path': str(self.qhelper), 'reviewed_quantizer_helper_sha256': QUANTIZER_HELPER_SHA,
                'global_budget_seconds': H.BUDGET_SECONDS, 'absolute_work_cutoff_utc': H.ABSOLUTE_CUTOFF.isoformat(),
                'cleanup_reserve_seconds': 15, 'planned_stages': planned_names(), 'stages': [],
                'original_B_root': H.B_ROOT, 'prior_quantizer_run': args.prior_quantizer_run,
                'certified_variant': 2, 'tiled_variant': 1, 'prepare_raw_cache_default_enabled': False,
                'quantizer_rne_enabled_both_builds': True, 'source_expected_commit': SOURCE_COMMIT,
                'source_expected_archive_sha256': SOURCE_SHA, 'source_expected_bytes': SOURCE_BYTES,
                'shapes': list(SHAPES), 'input_modes': {str(k): v for k,v in MODES.items()},
                'per_shape_order': list(ORDER), 'warmup': 100, 'rounds': 5, 'iters': 10, 'seed': 20261123,
                'input_fingerprints': {}, 'independent_process_runs_per_tag_shape': 2,
                'rounds_are_independent_samples': False, 'gpu_executed': False, 'automatic_retries': False,
                'training_smoke_executed': False, 'full_training_executed': False,
                'llm_convergence_validated': False, 'clock_or_power_changes': False,
                'packages_installed': False, 'services_stopped': False}

        def integrity(self):
            H.Runner.integrity(self)
            H.need(H.sha(HELPER) == HELPER_SHA and H.sha(self.qhelper) == QUANTIZER_HELPER_SHA
                   and H.sha(Path(__file__)) == self.own_hash, 'Frozen execution helper/runner changed')
            if self.binding_hash is not None:
                H.need(H.sha(self.args.source_archive) == SOURCE_SHA and H.sha(self.args.source_binding) == self.binding_hash,
                       'Frozen source archive/binding changed')

        def build_gate(self, tag):
            folder = self.run / ('build-' + tag); data = H.document(folder/'compile-evidence.json')
            enabled = tag == 'on'
            H.fields(data, {'schema_version': 2, 'passed': True, 'gpu_executed': False,
                'certified_enabled': True, 'certified_variant': 2, 'tiled_variant': 1,
                'quantizer_rne_enabled': True, 'quantizer_rne_backend': 'sm70_mx32_rne_warp_fp32_v1',
                'certified_prepare_raw_cache_enabled': enabled,
                'certified_prepare_raw_cache_backend': 'sm70_c2_row_raw_cache_k8192_v1' if enabled else None,
                'certified_prepare_raw_cache_max_k': 8192 if enabled else None,
                'certified_prepare_raw_cache_bytes_per_mx32': 20 if enabled else None}, 'OFF/ON compile manifest')
            H.need('release 12.8' in data.get('nvcc_version', ''), 'Compiler is not CUDA12.8')
            native = {k[len(H.TRAINING):]:v for k,v in self.expected.items() if k.startswith(H.TRAINING+'csrc/')}
            H.need(data.get('source_sha256') == native and data.get('fragment_mapping_sha256') == self.expected[H.MAPPING],
                   'Compile sources/mapping differ')
            jobs = data.get('jobs', [])
            H.need(len(jobs) == 5 and {j.get('name') for j in jobs} == set(H.ARTIFACTS)
                   and set(data.get('artifacts', {})) == set(H.ARTIFACTS.values()), 'Require all five native artifacts per build')
            for job in jobs:
                argv = job.get('argv', [])
                H.need(argv and argv[0] == H.NVCC and job.get('returncode') == 0
                       and '-DMXFP4_CERTIFIED_VARIANT=2' in argv and '-DMXFP4_TILED_VARIANT=1' in argv
                       and '--ftz=false' in argv and '-gencode=arch=compute_70,code=[sm_70,compute_70]' in argv,
                       'Compile command/variant/SM70/FTZ differs')
                defines = [a for a in argv if a.startswith('-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE')]
                H.need(defines == (['-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE=1'] if enabled else []),
                       'Cache macro differs from OFF/ON build label')
                if job['name'] == 'shared_library':
                    H.need(str(self.source/H.TRAINING/'csrc/mxfp4_quantize_rne.cu') in argv, 'RNE source missing from library')
                log = H.safe_name(job['log'])
                H.need('/' not in log and H.sha(folder/log) == job.get('log_sha256'), 'Compile log hash differs')
            for name, meta in data['artifacts'].items():
                path=folder/name; H.regular(path)
                H.need(meta == {'sha256':H.sha(path),'bytes':path.stat().st_size}, 'Artifact bytes/hash differ')
                self.binaries[str(path)] = meta['sha256']
            return {'passed':True,'gpu_executed':False,'tag':tag,'prepare_raw_cache_enabled':enabled,
                    'artifacts':data['artifacts'],'compile_manifest_sha256':H.sha(folder/'compile-evidence.json')}

        def await_rnev2(self):
            row={'name':'wait-rnev2-lock','status':'waiting','passed':False,'executed':False,
                 'gpu_executed':False,'started_at_utc':H.utc()}
            self.report['stages'].append(row); self.save(); fd=None; started=time.monotonic()
            try:
                prior=Path(self.args.prior_quantizer_run)
                H.need(str(Path(H.B_ROOT).resolve(strict=True))==H.B_ROOT,'Original B lock identity differs')
                lock='/tmp/mxfp4-framework-followup-'+str(os.geteuid())+'-'+hashlib.sha256(H.B_ROOT.encode()).hexdigest()[:20]+'.lock'
                row['same_B_lock_path']=lock
                fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
                info=os.fstat(fd); H.need(info.st_uid==os.geteuid() and stat.S_ISREG(info.st_mode),'Unsafe same-UID GPU lock')
                while time.monotonic()<self.deadline-15:
                    if (prior/'archive-manifest.json').is_file():
                        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError: pass
                        else:
                            completion,evidence=Q.archive_evidence(prior,'native')
                            H.fields(completion,{'status':'passed','passed':True,'source_and_binaries_unchanged':True,
                                'gpu_executed':True,'all_numerical_gates_passed_before_performance':True,
                                'source_commit':Q.SOURCE_COMMIT,'source_archive_sha256':Q.SOURCE_SHA,
                                'runner_sha256':QUANTIZER_HELPER_SHA},'Completed RNE-v2 prerequisite')
                            stages=completion.get('stages',[])
                            H.need(len(stages)==20 and {s.get('name') for s in stages}==set(Q.planned_names()),
                                   'RNE-v2 complete 20-stage set differs')
                            H.need(all(s.get('passed') is True and s.get('status')=='passed' for s in stages),
                                   'A prior RNE-v2 stage did not pass')
                            quant=H.document(prior/'results/check-quantize-cuda.json')
                            H.fields(quant,{'status':'passed','passed':True,'gpu_executed':True,
                                'completed_cases':135,'expected_cases':135,'all_attempted_work_synchronized':True},'Prior RNE135')
                            H.need(len(quant.get('cases',[]))==135 and all(c.get('passed') is True for c in quant['cases']),
                                   'Prior RNE135 case coverage incomplete')
                            counts=Q.walk_metrics(quant['cases'])
                            H.need(counts=={'byte_comparisons':147,'numerical_comparisons':306},'Prior RNE numerical content differs')
                            self.lock_fd,fd=fd,None
                            row.update(status='passed',passed=True,same_uid_lock_acquired=True,prior=evidence,
                                       previous_numerical_metric_counts=counts)
                            destination=self.results/'prior-rnev2'; destination.mkdir(mode=0o700)
                            shutil.copyfile(prior/'archive-manifest.json',destination/'archive-manifest.json')
                            shutil.copyfile(prior/'results/completion.json',destination/'completion.json')
                            return True
                    time.sleep(min(2,max(0,self.deadline-15-time.monotonic())))
                raise H.Blocked('Deadline before completed RNE-v2 and same B lock were available')
            except (Exception,KeyboardInterrupt) as error:
                row.update(status='blocked' if isinstance(error,H.Blocked) else 'failed',error_type=type(error).__name__,error=str(error))
                if isinstance(error,KeyboardInterrupt):self.deadline=time.monotonic()
                return False
            finally:
                if fd is not None:os.close(fd)
                row.update(finished_at_utc=H.utc(),wall_seconds=time.monotonic()-started);self.save()

        def execute(self):
            if self.deadline<=self.start+15:raise H.Blocked('Absolute/global deadline exhausted')
            H.need(Path(self.args.python).is_file() and os.access(self.args.python,os.X_OK),'Lexical venv Python is not executable')
            if not Path(H.NVCC).is_file():raise H.Blocked('CUDA12.8 nvcc unavailable')
            H.need(H.regular(self.args.source_archive,owned=True).st_size==SOURCE_BYTES,'Frozen source archive byte count differs')
            binding,self.expected=H.extract(Path(self.args.source_archive),Path(self.args.source_binding),self.source)
            H.fields(binding,{'source_commit':SOURCE_COMMIT,'source_sha256':SOURCE_SHA},'Frozen cache source binding')
            H.need(len(self.expected)==SOURCE_FILES and
                   self.expected.get(H.TRAINING+'csrc/check_mxfp4_prepare_cache.cu')==MICRO_SHA,'Frozen source/microchecker set differs')
            self.binding_hash=H.sha(self.args.source_binding)
            H.write_json(self.results/'source-binding.json',binding)
            shutil.copyfile(Path(__file__),self.results/'prepare-cache-experiment-runner.py')
            shutil.copyfile(HELPER,self.results/'reviewed-native-helper.py')
            shutil.copyfile(self.qhelper,self.results/'reviewed-rnev2-helper.py')
            self.report.update(source_commit=SOURCE_COMMIT,source_archive_sha256=SOURCE_SHA,source_archive_bytes=SOURCE_BYTES,
                               source_hashes_before=self.expected,source_binding_sha256=self.binding_hash)
            self.integrity()
            builds={}
            for tag in TAGS:
                command=[self.args.python,'-m','tools.mxfp4_training.build_cuda','--output-dir',str(self.run/('build-'+tag)),
                    '--nvcc',H.NVCC,'--with-certified','--with-benchmark','--with-quantizer-rne',
                    '--certified-variant','2','--tiled-variant','1']
                if tag=='on':command.append('--certified-prepare-raw-cache')
                builds[tag]=self.stage('build-'+tag,command,900,lambda rows,t=tag:self.build_gate(t))
            micro=self.run/'build-micro/check_mxfp4_prepare_cache';micro.parent.mkdir(mode=0o700)
            csrc=self.source/H.TRAINING/'csrc'
            def micro_build_gate(rows):
                H.regular(micro);self.binaries[str(micro)]=H.sha(micro)
                return {'passed':True,'gpu_executed':False,'binary_sha256':H.sha(micro),
                        'checker_source_sha256':MICRO_SHA,'actual_included_certified_sha256':H.sha(csrc/'mxfp4_gemm_certified.cu')}
            micro_built=self.stage('build-prepare-micro',[H.NVCC,'-O3','-std=c++17','--ftz=false','-lineinfo','-Xptxas=-v',
                '-gencode=arch=compute_70,code=[sm_70,compute_70]','-DMXFP4_CERTIFIED_VARIANT=2','-DMXFP4_TILED_VARIANT=1',
                '-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE=1',str(csrc/'check_mxfp4_prepare_cache.cu'),str(csrc/'mxfp4_gemm_tiled.cu'),
                '-lcublas','-o',str(micro)],900,micro_build_gate)
            self.report['binary_hashes_before_gpu']=dict(self.binaries)
            if not all(builds.values()) or not micro_built:raise H.Blocked('Both builds and actual preparation microcheck must compile')
            if not self.await_rnev2():raise H.Blocked('Completed RNE-v2/same B GPU serialization prerequisite failed')
            for tag in TAGS:
                for kind,filename,variant in (('baseline','check_mxfp4',None),('tiled','check_mxfp4_tiled',None),
                                              ('certified','check_mxfp4_certified',2)):
                    name=f'check-{tag}-{kind}'
                    if not self.stage(name,[self.run/('build-'+tag)/filename,'--device','0','--correctness-only'],600,
                                      lambda rows,v=variant:H.native_gate(rows,v),gpu=True):
                        raise H.Blocked('Native numerical control failed: '+name)
            if not self.stage('check-prepare-micro',[micro,'--device','0','--correctness-only'],600,micro_gate,gpu=True):
                raise H.Blocked('Actual cached-vs-original preparation/CPU-oracle gate failed')
            output=self.results/'torch-certified-on.json'
            if not self.stage('torch-certified-on',[self.args.python,'-m','tools.mxfp4_training.check_torch_certified',
                '--compile-evidence',str(self.manifest),'--expected-certified-variant','2','--output',str(output),
                '--device','cuda:0','--seed','20261123'],600,lambda rows:self.torch_gate(output,True),gpu=True,library=self.library):
                raise H.Blocked('Cache-ON certified Torch16 failed')
            for tag in TAGS:
                self.library=self.run/('build-'+tag)/'libmxfp4_sm70.so'
                self.manifest=self.run/('build-'+tag)/'compile-evidence.json'
                name='quantize-cuda-'+tag;output=self.results/(name+'.json')
                if not self.stage(name,[self.args.python,'-m','tools.mxfp4_training.check_quantize_cuda',
                    '--compile-evidence',str(self.manifest),'--output',str(output),'--device','cuda:0','--seed','20261123'],600,
                    lambda rows,p=output:self.quantizer_gate(p),gpu=True,library=self.library):
                    raise H.Blocked('Cache-'+tag+' CUDA RNE checker failed')
            self.report['all_numerical_gates_passed_before_performance']=True;self.save()
            for size in SHAPES:
                for index,tag in enumerate(ORDER):
                    name=f'bench-{size}-{index+1:02d}-{tag}';output=self.results/(name+'.jsonl')
                    benchmark=self.run/('build-'+tag)/'bench_mxfp4'
                    command=[benchmark,'--device','0','--m',str(size),'--n',str(size),'--k',str(size),
                        '--mode',MODES[size],'--seed','20261123','--warmup','100','--iters','10','--rounds','5',
                        '--samples','256','--output',str(output)]
                    def gate(rows,p=output,s=size,t=tag):
                        result=H.benchmark_gate(H.json_lines(p),2,s,self.report['input_fingerprints'],MODES[s])
                        result.update(prepare_raw_cache_enabled=t=='on',build_tag=t,
                                      output_sha256=H.sha(p),primary_scope='whole_certified_pipeline')
                        return result
                    if not self.stage(name,command,900,gate,gpu=True):
                        raise H.Blocked('OFF/ON benchmark control/correctness failed; comparisons stopped')

        def finish(self,error=None):
            self.cleanup()
            if error is not None:self.report.update(error_type=type(error).__name__,error=str(error))
            seen={s['name'] for s in self.report['stages']}
            for name in planned_names():
                if name not in seen:self.block(name,'Prerequisite/control/deadline prevented execution: '+str(error),
                                              not name.startswith('build-') and name!='wait-rnev2-lock')
            try:
                self.integrity()
                self.report['source_hashes_after']=H.tree_hashes(self.source) if self.expected is not None else None
                self.report['source_and_binaries_unchanged']=self.expected is not None
                self.report['binary_hashes_after_gpu']={p:H.sha(p) for p in self.binaries}
            except Exception as changed:
                self.report.update(integrity_error=str(changed),source_and_binaries_unchanged=False);error=changed
            try:
                for build in sorted(self.run.glob('build-*')):
                    target=self.results/'build-evidence'/build.name;target.mkdir(parents=True,exist_ok=True)
                    for path in sorted(build.iterdir()):
                        if path.suffix in ('.json','.log') and path.is_file() and not path.is_symlink():shutil.copyfile(path,target/path.name)
                statuses={s['status'] for s in self.report['stages']}
                status='failed' if 'failed' in statuses or (error is not None and not isinstance(error,H.Blocked)) else (
                    'blocked' if 'blocked' in statuses or error is not None else 'passed')
                gpu=[s.get('gpu_executed') for s in self.report['stages'] if s.get('executed')]
                self.report.update(status=status,passed=status=='passed',finished_at_utc=H.utc(),
                    wall_seconds=time.monotonic()-self.start,gpu_executed=True if True in gpu else (None if None in gpu else False))
                H.need(len(self.report['stages'])==26 and {s['name'] for s in self.report['stages']}==set(planned_names()),
                       'Final independent26-stage plan differs')
                self.report['result_files_sha256']={k:v for k,v in H.tree_hashes(self.results).items() if k!='completion.json'}
                self.save()
                H.need(sum(p.stat().st_size for p in self.results.rglob('*') if p.is_file())<=H.MAX_ARCHIVE_BYTES,'Results archive exceeds bound')
                archive=self.run/'results.tar.gz'
                with tarfile.open(archive,'x:gz') as target:target.add(self.results,arcname='results',recursive=True)
                receipt={'status':status,'passed':status=='passed','results_archive':str(archive),'sha256':H.sha(archive),
                    'bytes':archive.stat().st_size,'completion_sha256':H.sha(self.results/'completion.json'),
                    'source_commit':self.report.get('source_commit'),'source_archive_sha256':self.report.get('source_archive_sha256'),
                    'gpu_executed':self.report['gpu_executed'],'runner_sha256':self.own_hash,
                    'reviewed_native_helper_sha256':HELPER_SHA,'reviewed_quantizer_helper_sha256':QUANTIZER_HELPER_SHA}
                H.write_json(self.run/'archive-manifest.json',receipt);print(json.dumps(receipt),flush=True)
                return {'passed':0,'blocked':2,'failed':1}[status]
            finally:
                if self.lock_fd is not None:fcntl.flock(self.lock_fd,fcntl.LOCK_UN);os.close(self.lock_fd);self.lock_fd=None
    return CacheRunner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-archive','source-binding','run-directory','python'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--prior-quantizer-run',default=DEFAULT_PRIOR)
    args=parser.parse_args()
    for name in ('source_archive','source_binding','run_directory','python','prior_quantizer_run'):
        if not os.path.isabs(getattr(args,name)):parser.error('Every path must be explicitly absolute')
    global H,Q
    H=reviewed_module(HELPER,HELPER_SHA,'mxfp4_frozen_native_helper')
    prior=Path(args.prior_quantizer_run)
    H.need(prior.is_dir() and not prior.is_symlink() and prior.stat().st_uid==os.geteuid(),'Prior RNE-v2 directory must exist and be owned')
    prior=prior.resolve(strict=True);args.prior_quantizer_run=str(prior)
    Q=reviewed_module(prior/'results/quantizer-experiment-runner.py',QUANTIZER_HELPER_SHA,'mxfp4_frozen_rnev2_helper');Q.H=H
    run=Path(args.run_directory)
    H.need(run.parent.is_dir() and not run.exists() and not run.is_symlink(),'New run directory must not exist')
    run=run.parent.resolve(strict=True)/run.name
    H.need(run!=prior and run not in prior.parents and prior not in run.parents,'New and prior run directories overlap')
    H.need(str(run)!=H.B_ROOT and Path(H.B_ROOT) not in run.parents and run not in Path(H.B_ROOT).parents,'Do not modify original B run')
    args.framework_run=str(prior)
    run.mkdir(mode=0o700,exist_ok=False);runner=create_runner_class()(args,run)
    def interrupted(signum,frame):
        for handled in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(handled,signal.SIG_IGN)
        raise KeyboardInterrupt('Prepare cache runner received signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupted)
    error=None
    try:runner.execute()
    except (Exception,KeyboardInterrupt) as caught:error=caught
    return runner.finish(error)


if __name__=='__main__':raise SystemExit(main())
