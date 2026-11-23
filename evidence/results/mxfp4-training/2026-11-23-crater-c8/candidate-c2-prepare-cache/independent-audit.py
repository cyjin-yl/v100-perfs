#!/usr/bin/env python3
"""Read-only audit of an actual C2 prepare-cache OFF/ON result archive.

Only derived output files are written. No CUDA imports, remote access, test
fixtures, result synthesis, or extraction into the source/run directories.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile
import types

SOURCE_COMMIT='c581389551be2233ba23404fa72c471cc4c6f961'
SOURCE_SHA='a7062abcfa1b9263d7a7bf0d282f2de1b51cfa279939f3b969eacb543f5417c0'
SOURCE_BYTES=146431
RUNNER_SHA='65e23628156a4430421cd6ad21359a62e4f2ab922f9daf58ee7d1621b5ecdde7'
NATIVE_HELPER_SHA='f7f45628bcf7616a7e7915823c77f2f75602cff66be29bf640c5ec3be8ecce1e'
RNE_HELPER_SHA='96d3db40d25426370cfc8542dff908edf00a385c0055019c727c9c005fb8a548'
TRAINING='tools/mxfp4_training/'
MAPPING='evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp'
MODES={2048:'random_block_scales',4096:'random_block_scales',8192:'random_unit_scale'}
ORDER=('off','on','on','off')
LIMIT=1024**3


class AuditError(ValueError):pass

def need(value,message):
    if not value:raise AuditError(message)

def digest(data):return hashlib.sha256(data).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def pairs(values):
    result={}
    for key,value in values:
        need(key not in result,'Duplicate JSON key: '+key);result[key]=value
    return result

def load(data):
    def invalid(value):raise AuditError('Non-finite JSON number: '+value)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=invalid)

def fields(data,expected,label):
    for key,value in expected.items():
        need(data.get(key)==value and (not isinstance(value,bool) or data.get(key) is value),label+': '+key)

def safe(name):
    path=PurePosixPath(name)
    need(name and '\\' not in name and not path.is_absolute() and '..' not in path.parts
         and '.' not in name.split('/') and str(path)==name.rstrip('/'),'Unsafe archive name: '+name)
    return str(path)

class Bundle:
    def __init__(self,path,expected_hash):
        self.path=Path(path);need(self.path.is_file() and not self.path.is_symlink(),'Archive must be a regular file')
        self.bytes=self.path.stat().st_size;need(self.bytes<=LIMIT,'Archive exceeds size bound')
        self.sha256=sha(self.path);need(self.sha256==expected_hash,'Archive SHA mismatch')
        self.files={};self.hashes={};seen=set();expanded=0
        with tarfile.open(self.path,'r:*') as tar:
            members=tar.getmembers();need(len(members)<10000,'Too many archive members')
            for member in members:
                name=safe(member.name);need(name not in seen,'Repeated tar member');seen.add(name)
                need(member.isdir() or member.isreg(),'Links/special tar members forbidden')
                need(not getattr(member,'sparse',None),'Sparse tar members forbidden')
                if member.isdir():continue
                expanded+=member.size;need(0<=member.size<=LIMIT and expanded<=LIMIT,'Expanded archive exceeds bound')
                data=tar.extractfile(member).read();need(len(data)==member.size,'Short tar member')
                self.files[name]=data;self.hashes[name]=digest(data)
    def document(self,name):need(name in self.files,'Missing artifact: '+name);return load(self.files[name])
    def rows(self,name):
        need(name in self.files,'Missing JSONL artifact: '+name)
        return [load(line) for line in self.files[name].decode().splitlines() if line.startswith('{')]


def reviewed(data,expected,name):
    need(digest(data)==expected,'Reviewed helper/runner differs: '+name)
    module=types.ModuleType(name);module.__file__='<archived:'+name+'>';sys.modules[name]=module
    exec(compile(data,module.__file__,'exec'),module.__dict__)
    return module

def close(a,b,label):
    need(isinstance(a,(float,int)) and isinstance(b,(float,int)) and math.isfinite(a) and math.isfinite(b)
         and math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-7),label+' printed/recomputed mismatch')

def source_subset(source,names,all_native=False,mapping=False):
    result={name:source.hashes[TRAINING+name] for name in names}
    if all_native:result.update({k[len(TRAINING):]:v for k,v in source.hashes.items() if k.startswith(TRAINING+'csrc/')})
    if mapping:result['fragment_mapping.hpp']=source.hashes[MAPPING]
    return result


def summarize_telemetry(benches):
    snapshots=[]
    for run in benches:
        for field,phase in (('telemetry','environment_before_case'),
                            ('telemetry_before','after_all_backend_warmups'),
                            ('telemetry_after','after_all_backend_timings_and_validation')):
            snapshot=run['environment'][field] if field=='telemetry' else run['timed_correctness_and_all_job_timings'][field]
            need(isinstance(snapshot,dict) and type(snapshot.get('nvml_available')) is bool,'Invalid NVML snapshot')
            for key in ('sm_clock_mhz','memory_clock_mhz','power_w','power_limit_w'):
                value=snapshot.get(key)
                need(value is None or (type(value) in (int,float) and math.isfinite(value) and value>=0),
                     'Invalid NVML metric: '+key)
            snapshots.append({'stage':run['stage'],'size':run['size'],'input_mode':run['input_mode'],
                              'tag':run['tag'],'field':field,'phase':phase,'values':snapshot})
    def ranges(selected):
        result={'snapshot_count':len(selected),
                'nvml_available_count':sum(s['values']['nvml_available'] is True for s in selected)}
        for key in ('sm_clock_mhz','memory_clock_mhz','power_w','power_limit_w'):
            values=[s['values'][key] for s in selected if type(s['values'].get(key)) in (int,float)]
            result[key]={'count':len(values),'min':min(values),'max':max(values),'values':values} if values else None
        return result
    boundaries=[s for s in snapshots if s['field']!='telemetry']
    return {'snapshot_count':len(snapshots),'raw_snapshots':snapshots,
            'all_snapshots':ranges(snapshots),'benchmark_boundary_snapshots':ranges(boundaries),
            'by_phase':{phase:ranges([s for s in snapshots if s['phase']==phase]) for phase in
                        ('environment_before_case','after_all_backend_warmups','after_all_backend_timings_and_validation')},
            'benchmark_boundaries_by_shape':{str(size):ranges([s for s in boundaries if s['size']==size]) for size in MODES},
            'benchmark_boundaries_by_shape_and_tag':{
                str(size):{tag:ranges([s for s in boundaries if s['size']==size and s['tag']==tag]) for tag in ('off','on')}
                for size in MODES},
            'continuous_sampling':False,'utilization_recorded':False,'temperature_recorded':False,
            'scope':'NVML snapshots cover each benchmark process; before is after all six backends warm up, after is after all backend timings and CPU output validation. They are not per-C2-interval measurements.',
            'interpretation':'Observed snapshot extrema do not establish exclusive GPU use, continuous clocks, or active-kernel average power.'}


def expected_command(name,run,python,H):
    csrc=run+'/source/'+TRAINING+'csrc/'
    if name in ('build-off','build-on'):
        tag=name[6:]
        value=[python,'-m','tools.mxfp4_training.build_cuda','--output-dir',run+'/'+name,'--nvcc',H.NVCC,
               '--with-certified','--with-benchmark','--with-quantizer-rne','--certified-variant','2','--tiled-variant','1']
        return value+(['--certified-prepare-raw-cache'] if tag=='on' else [])
    if name=='build-prepare-micro':return [H.NVCC,'-O3','-std=c++17','--ftz=false','-lineinfo','-Xptxas=-v',
        '-gencode=arch=compute_70,code=[sm_70,compute_70]','-DMXFP4_CERTIFIED_VARIANT=2','-DMXFP4_TILED_VARIANT=1',
        '-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE=1',csrc+'check_mxfp4_prepare_cache.cu',csrc+'mxfp4_gemm_tiled.cu',
        '-lcublas','-o',run+'/build-micro/check_mxfp4_prepare_cache']
    if name.startswith(('check-off-','check-on-')):
        _,tag,kind=name.split('-');exe={'baseline':'check_mxfp4','tiled':'check_mxfp4_tiled','certified':'check_mxfp4_certified'}[kind]
        return [run+'/build-'+tag+'/'+exe,'--device','0','--correctness-only']
    if name=='check-prepare-micro':return [run+'/build-micro/check_mxfp4_prepare_cache','--device','0','--correctness-only']
    if name=='torch-certified-on':return [python,'-m','tools.mxfp4_training.check_torch_certified',
        '--compile-evidence',run+'/build-on/compile-evidence.json','--expected-certified-variant','2',
        '--output',run+'/results/'+name+'.json','--device','cuda:0','--seed','20261123']
    if name.startswith('quantize-cuda-'):
        tag=name.rsplit('-',1)[1]
        return [python,'-m','tools.mxfp4_training.check_quantize_cuda','--compile-evidence',run+'/build-'+tag+'/compile-evidence.json',
                '--output',run+'/results/'+name+'.json','--device','cuda:0','--seed','20261123']
    if name.startswith('bench-'):
        _,size,index,tag=name.split('-');mode=MODES[int(size)]
        return [run+'/build-'+tag+'/bench_mxfp4','--device','0','--m',size,'--n',size,'--k',size,'--mode',mode,
                '--seed','20261123','--warmup','100','--iters','10','--rounds','5','--samples','256',
                '--output',run+'/results/'+name+'.jsonl']
    raise AuditError('Unknown command stage: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--source-archive',type=Path,required=True)
    parser.add_argument('--source-binding',type=Path,required=True)
    parser.add_argument('--expected-results-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--markdown',type=Path)
    args=parser.parse_args()
    if args.output.exists() or (args.markdown and args.markdown.exists()):parser.error('Preserve existing evidence; choose new output paths')
    result={'schema_version':2,'purpose':'independent actual C2 prepare-cache OFF/ON archive audit',
            'auditor_sha256':sha(Path(__file__)),'audit_started_at_utc':datetime.now(timezone.utc).isoformat(),
            'revision_note':'Extract existing native benchmark NVML boundary snapshots; no raw inputs, numerical gates, or timing calculations changed.',
            'audit_passed':False,'audit_status':'pending','gpu_executed_by_auditor':False,'performance_rows':[]}
    code=1
    try:
        source=Bundle(args.source_archive,SOURCE_SHA);need(source.bytes==SOURCE_BYTES and len(source.files)==30,'Frozen source size/count differs')
        binding=load(args.source_binding.read_bytes())
        fields(binding,{'source_commit':SOURCE_COMMIT,'source_sha256':SOURCE_SHA},'Frozen source binding')
        members=binding.get('members',[]);need(len(members)==30,'Source member list incomplete')
        need({safe(m['name']):m['sha256'] for m in members}==source.hashes,'Source member hashes differ')
        archive=Bundle(args.directory/'results.tar.gz',args.expected_results_sha256)
        receipt=load((args.directory/'archive-manifest.json').read_bytes());completion=archive.document('results/completion.json')
        fields(receipt,{'sha256':archive.sha256,'bytes':archive.bytes,'runner_sha256':RUNNER_SHA,
            'reviewed_native_helper_sha256':NATIVE_HELPER_SHA,'reviewed_quantizer_helper_sha256':RNE_HELPER_SHA},'Result receipt')
        need(archive.hashes['results/completion.json']==receipt.get('completion_sha256'),'Completion SHA differs from receipt')
        local_completion=args.directory/'results/completion.json'
        if local_completion.exists():need(sha(local_completion)==receipt['completion_sha256'],'Extracted completion differs')
        need(all(k.startswith('results/') for k in archive.files),'Out-of-root result file')
        files={k[len('results/'):]:v for k,v in archive.hashes.items() if k!='results/completion.json'}
        need(files==completion.get('result_files_sha256'),'Result members differ from exact completion hash map')
        status=completion.get('status');need(status in ('passed','failed','blocked'),'Experiment is not terminal')
        fields(receipt,{'status':status,'passed':status=='passed'},'Experiment receipt state')
        fields(completion,{'passed':status=='passed','source_expected_commit':SOURCE_COMMIT,
            'source_expected_archive_sha256':SOURCE_SHA,'source_expected_bytes':SOURCE_BYTES,
            'runner_sha256':RUNNER_SHA,'reviewed_native_helper_sha256':NATIVE_HELPER_SHA,
            'reviewed_quantizer_helper_sha256':RNE_HELPER_SHA},'Execution binding')
        materialized=completion.get('source_commit') is not None
        if materialized:
            fields(completion,{'source_commit':SOURCE_COMMIT,'source_archive_sha256':SOURCE_SHA,'source_archive_bytes':SOURCE_BYTES},'Materialized source')
            need(completion.get('source_hashes_before')==source.hashes,'Initial extracted source differs')
            need(archive.document('results/source-binding.json')==binding,'Archived source binding differs')
        H=reviewed(archive.files['results/reviewed-native-helper.py'],NATIVE_HELPER_SHA,'audit_native_helper')
        Q=reviewed(archive.files['results/reviewed-rnev2-helper.py'],RNE_HELPER_SHA,'audit_rnev2_helper');Q.H=H
        C=reviewed(archive.files['results/prepare-cache-experiment-runner.py'],RUNNER_SHA,'audit_cache_runner');C.H=H;C.Q=Q
        planned=C.planned_names();stages=completion.get('stages',[])
        need(len(planned)==26 and len(stages)==26 and len({s.get('name') for s in stages})==26
             and {s.get('name') for s in stages}==set(planned),'Independent26-stage set differs')
        by_name={s['name']:s for s in stages};run=completion['run_directory'];python=completion['python_execution_path']
        numerical={};builds={};quant={};benches=[];fingerprints={};micro=None;torch16=None
        def passed(name):return by_name[name].get('status')=='passed'
        for name in planned:
            stage=by_name[name];state=stage.get('status')
            need(state in ('passed','failed','blocked') and stage.get('passed') is (state=='passed'),'Stage terminal state invalid: '+name)
            if name=='wait-rnev2-lock':
                if state=='passed':
                    need(stage.get('same_uid_lock_acquired') is True and stage.get('executed') is False,'Prior serialization proof missing')
                    prior_receipt=archive.document('results/prior-rnev2/archive-manifest.json')
                    prior_completion=archive.document('results/prior-rnev2/completion.json')
                    fields(prior_completion,{'status':'passed','passed':True,'source_commit':Q.SOURCE_COMMIT,
                        'source_archive_sha256':Q.SOURCE_SHA,'runner_sha256':RNE_HELPER_SHA},'Copied prior RNE-v2')
                    need(archive.hashes['results/prior-rnev2/completion.json']==prior_receipt['completion_sha256'],'Copied prior completion hash differs')
                    fields(stage['prior'],{'status':'passed','receipt_sha256':archive.hashes['results/prior-rnev2/archive-manifest.json'],
                           'completion_sha256':prior_receipt['completion_sha256'],'archive_sha256':prior_receipt['sha256']},'Prior linked receipt')
                continue
            if stage.get('executed'):
                need(stage.get('argv')==expected_command(name,run,python,H) and stage.get('cwd')==run+'/source',
                     'Stage command/source path binding differs: '+name)
                prefix='results/stages/'+name+'/'
                need(archive.document(prefix+'stage.json')==stage,'Stage record differs from final completion: '+name)
                expected_logs={'stdout.log':archive.hashes[prefix+'stdout.log'],'stderr.log':archive.hashes[prefix+'stderr.log']}
                need(stage.get('evidence_sha256')==expected_logs,'Stage stdout/stderr hashes differ: '+name)
            if state!='passed':continue
            need(stage.get('executed') is True and stage.get('returncode')==0 and stage.get('validation',{}).get('passed') is True,
                 'Passing executable stage lacks actual successful execution/gate: '+name)
            rows=archive.rows('results/stages/'+name+'/stdout.log')
            if name in ('build-off','build-on'):
                tag=name[6:];enabled=tag=='on';prefix='results/build-evidence/'+name+'/'
                data=archive.document(prefix+'compile-evidence.json')
                fields(data,{'passed':True,'gpu_executed':False,'certified_enabled':True,'certified_variant':2,'tiled_variant':1,
                    'quantizer_rne_enabled':True,'certified_prepare_raw_cache_enabled':enabled},'Actual compile manifest')
                need(data.get('source_sha256')=={k[len(TRAINING):]:v for k,v in source.hashes.items() if k.startswith(TRAINING+'csrc/')}
                     and data.get('fragment_mapping_sha256')==source.hashes[MAPPING],'Compiled native source/mapping mismatch')
                need('release 12.8' in data.get('nvcc_version',''),'Compiler version differs')
                jobs=data.get('jobs',[]);need(len(jobs)==5 and {j.get('name') for j in jobs}==set(H.ARTIFACTS),'Compilation job set differs')
                need(set(data.get('artifacts',{}))==set(H.ARTIFACTS.values()),'Compiled artifact set differs')
                for job in jobs:
                    argv=job.get('argv',[])
                    need(job.get('returncode')==0 and argv and argv[0]==H.NVCC and '-DMXFP4_CERTIFIED_VARIANT=2' in argv
                         and '--ftz=false' in argv and '-gencode=arch=compute_70,code=[sm_70,compute_70]' in argv,'Compiler flags/status invalid')
                    need([a for a in argv if a.startswith('-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE')]
                         ==(['-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE=1'] if enabled else []),'Cache OFF/ON macro differs')
                    need('/' not in safe(job['log']) and archive.hashes[prefix+job['log']]==job['log_sha256'],'Compile log hash differs')
                need(stage['validation']['compile_manifest_sha256']==archive.hashes[prefix+'compile-evidence.json'],'Build gate manifest differs')
                builds[tag]={'manifest':data,'manifest_sha256':archive.hashes[prefix+'compile-evidence.json']}
            elif name=='build-prepare-micro':
                fields(stage['validation'],{'checker_source_sha256':C.MICRO_SHA,
                    'actual_included_certified_sha256':source.hashes[TRAINING+'csrc/mxfp4_gemm_certified.cu']},'Actual microcheck source')
            elif name.startswith(('check-off-','check-on-')):
                kind=name.rsplit('-',1)[1];numerical[name]=H.native_gate(rows,2 if kind=='certified' else None)
            elif name=='check-prepare-micro':
                micro=C.micro_gate(rows)
                result['micro_cases']=[r for r in rows if r.get('type')=='prepare_cache_correctness']
            elif name in ('torch-certified-on','quantize-cuda-off','quantize-cuda-on'):
                tag='off' if name.endswith('-off') else 'on';need(tag in builds,'Torch suite without passing build')
                data=archive.document('results/'+name+'.json');meta=builds[tag]['manifest']['artifacts']['libmxfp4_sm70.so']
                fields(data,{'status':'passed','passed':True,'gpu_executed':True,'all_attempted_work_synchronized':True,
                             'source_unchanged_during_check':True},'Actual Torch/RNE suite')
                fields(data['library'],{'path':run+'/build-'+tag+'/libmxfp4_sm70.so',**meta,
                    'compile_evidence_sha256':builds[tag]['manifest_sha256']},'Torch/RNE library binding')
                need(stage.get('library_sha256_before')==stage.get('library_sha256_after')==meta['sha256'],'Executed Torch library changed')
                count=16 if name=='torch-certified-on' else 135
                fields(data,{'completed_cases':count},'Actual Torch/RNE case count')
                cases=data.get('cases',[]);need(len(cases)==count and all(c.get('passed') is True for c in cases),'Incomplete Torch/RNE cases')
                counts=Q.walk_metrics(cases)
                if name=='torch-certified-on':
                    want=source_subset(source,['check_torch_certified.py','check_torch_cuda.py','reference.py','torch_ops.py'],True,True)
                    fields(data['library'],{'compiled_variant':2,'native_sources_match':True,'fragment_mapping_matches':True},'Certified Torch provenance')
                    torch16={'cases':16,'metric_counts':counts,'report_sha256':archive.hashes['results/'+name+'.json']}
                else:
                    want=source_subset(source,['check_quantize_cuda.py','reference.py','torch_ops.py','build_cuda.py'],True)
                    fields(data,{'expected_cases':135,'tested_gemm_backends':['cuda','cuda_tiled','cuda_certified']},'RNE coverage')
                    need(counts=={'byte_comparisons':147,'numerical_comparisons':306},'RNE135 actual comparison counts differ')
                    lazy=[c for c in cases if c.get('name')=='lazy_negative_input']
                    need(len(lazy)==1 and lazy[0].get('torch_control_lazy_view_resolved') is True,'Lazy-negative control accommodation missing')
                    quant[tag]={'cases':135,'metric_counts':counts,'report_sha256':archive.hashes['results/'+name+'.json'],
                                'all_cases':cases,'library_sha256':meta['sha256']}
                need(data.get('source_sha256')==want,'Torch/RNE source set differs')
            elif name.startswith('bench-'):
                _,shape,index,tag=name.split('-');size=int(shape)
                output='results/'+name+'.jsonl';records=archive.rows(output)
                need(records==rows,'Benchmark stdout/output differ; do not count duplicates as runs')
                value=H.benchmark_gate(records,2,size,fingerprints,MODES[size])
                need(tag in builds,'Benchmark lacks a passing matching build')
                binary=builds[tag]['manifest']['artifacts']['bench_mxfp4']['sha256']
                need(stage.get('binary_sha256_before')==stage.get('binary_sha256_after')==binary,'Executed benchmark binary changed')
                timed=H.one(records,'benchmark');initial=H.one(records,'correctness');job=value['certified_timing']
                wall=job['synchronized_wall_round_ms'];event=job['cuda_event_round_ms'];flops=2*size**3
                need(job['round_ms']==wall,'Certified primary raw timings are not wall time')
                close(job['median_ms'],statistics.median(wall),'Primary wall median')
                close(job['synchronized_wall_median_ms'],statistics.median(wall),'Wall median')
                close(job['cuda_event_median_ms'],statistics.median(event),'Event median')
                close(job['useful_tflops'],flops/(statistics.median(wall)*1e9),'Wall TFLOPS')
                close(job['cuda_event_useful_tflops'],flops/(statistics.median(event)*1e9),'Event TFLOPS')
                benches.append({'stage':name,'size':size,'input_mode':MODES[size],'tag':tag,'abba_position':int(index),
                    'input_fnv1a64':value['input_fnv1a64'],'output_sha256':archive.hashes[output],
                    'binary_sha256':binary,'wall_round_ms':wall,'event_round_ms':event,
                    'initial_certificate':initial['certificate'],'final_certificate':timed['certificate_after_timing'],
                    'environment':H.one(records,'environment'),
                    'initial_correctness':initial,'timed_correctness_and_all_job_timings':timed,
                    'initial_full_output_values_per_comparison':size**2,'final_actual_timed_certified_output_values':size**2})
        unchanged=completion.get('source_and_binaries_unchanged') is True
        if unchanged:
            need(completion.get('source_hashes_after')==source.hashes,'Final source hashes differ')
            need(completion.get('binary_hashes_before_gpu')==completion.get('binary_hashes_after_gpu'),'Final compiled binary map differs')
            binaries=completion['binary_hashes_after_gpu']
            for tag,build in builds.items():
                for name,meta in build['manifest']['artifacts'].items():
                    need(binaries.get(run+'/build-'+tag+'/'+name)==meta['sha256'],'Binary hash map differs from compile manifest')
            if passed('build-prepare-micro'):
                need(binaries.get(run+'/build-micro/check_mxfp4_prepare_cache')==by_name['build-prepare-micro']['validation']['binary_sha256'],
                     'Micro binary hash map differs')
        common=(unchanged and materialized and len(builds)==2 and len(numerical)==6 and micro is not None
                and torch16 is not None and set(quant)=={'off','on'} and passed('wait-rnev2-lock')
                and completion.get('all_numerical_gates_passed_before_performance') is True)
        comparisons=[]
        for size,mode in MODES.items():
            selected=[r for r in benches if r['size']==size]
            complete=(common and len(selected)==4 and sorted(r['abba_position'] for r in selected)==[1,2,3,4]
                      and [r['tag'] for r in sorted(selected,key=lambda r:r['abba_position'])]==list(ORDER))
            for tag in ('off','on'):
                runs=[r for r in selected if r['tag']==tag]
                row={'shape':[size]*3,'input_mode':mode,'tag':tag,'eligible':bool(complete),
                     'independent_process_runs':len(runs),'rounds_per_run':5,'pooled_round_count':10 if complete else None,
                     'raw_process_runs':runs,'pooled_wall_median_ms':None,'pooled_event_median_ms':None,
                     'wall_useful_tflops':None,'event_useful_tflops':None}
                if complete:
                    wall=[v for run_data in runs for v in run_data['wall_round_ms']]
                    event=[v for run_data in runs for v in run_data['event_round_ms']]
                    need(len(wall)==len(event)==10,'Pooled rounds incomplete')
                    row.update(pooled_wall_median_ms=statistics.median(wall),pooled_event_median_ms=statistics.median(event),
                        wall_useful_tflops=2*size**3/(statistics.median(wall)*1e9),
                        event_useful_tflops=2*size**3/(statistics.median(event)*1e9),
                        pooled_wall_round_ms=wall,pooled_event_round_ms=event)
                result['performance_rows'].append(row)
            a,b=result['performance_rows'][-2:]
            comparisons.append({'shape':[size]*3,'input_mode':mode,'eligible':bool(complete),
                'on_over_off_throughput_ratio':a['pooled_wall_median_ms']/b['pooled_wall_median_ms'] if complete else None,
                'on_wall_latency_reduction_percent':100*(1-b['pooled_wall_median_ms']/a['pooled_wall_median_ms']) if complete else None})
        need(not fingerprints or fingerprints==completion.get('input_fingerprints'),'Final packed fingerprint map differs')
        if status=='passed':need(common and len(benches)==12 and all(r['eligible'] for r in result['performance_rows']),
                                 'Experiment claims complete pass without all numerical/performance evidence')
        result.update(audit_passed=True,audit_status='verified',experiment_status=status,experiment_passed=status=='passed',
            result_archive={'path':str(args.directory/'results.tar.gz'),'sha256':archive.sha256,'bytes':archive.bytes,
                            'verified_hashed_result_files':len(files),'completion_sha256':receipt['completion_sha256']},
            source={'commit':SOURCE_COMMIT,'archive_sha256':SOURCE_SHA,'archive_bytes':SOURCE_BYTES,'verified_files':30},
            stage_counts={s:sum(x['status']==s for x in stages) for s in ('passed','failed','blocked')},
            stage_statuses=[{k:s.get(k) for k in ('name','status','passed','executed','gpu_executed','returncode','error','reason')} for s in stages],
            complete_numerical_gate_passed=bool(common),native_suites=numerical,prepare_micro_summary=micro,
            torch_certified16=torch16,quantizer135=quant,comparison=comparisons,input_fingerprints=fingerprints,
            telemetry=summarize_telemetry(benches),
            build_manifest_hashes={k:v['manifest_sha256'] for k,v in builds.items()},
            library_hashes={k:v['manifest']['artifacts']['libmxfp4_sm70.so']['sha256'] for k,v in builds.items()},
            binary_local_rehash_scope='Binaries are not in the result archive; their recorded before/after hashes are cross-bound to actual compile manifests and execution stages.',
            timing_scope='complete native certified pipeline including preparation/certificate/GEMM/postscale; primary synchronized wall clock',
            limitations=['Two process runs per tag/shape, five correlated rounds each; ten pooled rounds are not ten independent trials.',
                         '8192 uses random_unit_scale; 2048/4096 use random_block_scales.',
                         'This native packed-input benchmark excludes FP32 quantization, Python overhead and training.',
                         'Native benchmark records contain environment and before/after NVML snapshots; no continuous clock, utilization or temperature trace is recorded. The after snapshot follows all backend timings and CPU validation, so snapshot power is not active-C2 average power.',
                         'All three observed ON/OFF throughput gains are below 1%; with two process runs per tag/shape and changing 8192 boundary clocks, they are descriptive observations and do not justify changing the default OFF setting.',
                         'Kernel-path instrumentation is absent; public preparation routing is supported by the shared host guard and actual source binding.'])
        code=0
    except Exception as error:
        result.update(audit_status='inconsistent_or_incomplete',error_type=type(error).__name__,error=str(error))
        for row in result['performance_rows']:
            row.update(eligible=False,pooled_wall_median_ms=None,pooled_event_median_ms=None,wall_useful_tflops=None,event_useful_tflops=None)
        result['comparison']=[]
    result['audit_finished_at_utc']=datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    if args.markdown:
        lines=['# Independent C2 preparation-cache audit','',f"Audit: **{result['audit_status']}**.",'']
        if result['audit_passed']:
            lines += [f"Actual experiment: **{result['experiment_status']}**; stage counts: {result['stage_counts']}.",'',
                      '| Shape | Packed input mode | OFF wall ms / TFLOPS | ON wall ms / TFLOPS | ON/OFF throughput |',
                      '|---|---|---:|---:|---:|']
            for comparison in result['comparison']:
                size=comparison['shape'][0];rows=[r for r in result['performance_rows'] if r['shape'][0]==size]
                if comparison['eligible']:
                    a,b=rows;lines.append(f"| {size}³ | {comparison['input_mode']} | {a['pooled_wall_median_ms']:.6f} / {a['wall_useful_tflops']:.4f} | {b['pooled_wall_median_ms']:.6f} / {b['wall_useful_tflops']:.4f} | {comparison['on_over_off_throughput_ratio']:.5f}× |")
                else:lines.append(f"| {size}³ | {comparison['input_mode']} | unavailable | unavailable | unavailable |")
            lines += ['', 'All raw per-process rounds, full correctness records, certificate submissions and source/binary bindings are retained in the JSON.','']
            telemetry=result['telemetry']
            lines += ['## Recorded NVML boundaries','',
                      f"The native records contain {telemetry['snapshot_count']} snapshots: {telemetry['by_phase']['environment_before_case']['snapshot_count']} environment snapshots and {telemetry['benchmark_boundary_snapshots']['snapshot_count']} before/after benchmark snapshots. All are retained in the JSON.",'',
                      '| Shape | Before/after SM clock MHz | Memory clock MHz | Power snapshot W | Reported limit W |',
                      '|---|---:|---:|---:|---:|']
            for size,data in telemetry['benchmark_boundaries_by_shape'].items():
                def span(key):
                    value=data[key]
                    return 'unavailable' if value is None else f"{value['min']:g}–{value['max']:g}"
                lines.append(f"| {size}³ | {span('sm_clock_mhz')} | {span('memory_clock_mhz')} | {span('power_w')} | {span('power_limit_w')} |")
            lines += ['',telemetry['scope'],telemetry['interpretation'],'','## Limits','']
            lines += ['- '+value for value in result['limitations']]
        else:lines += [result.get('error','Audit did not complete.'),'','No performance value is eligible.']
        args.markdown.parent.mkdir(parents=True,exist_ok=True)
        with args.markdown.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'audit_status':result['audit_status'],'audit_passed':result['audit_passed'],'output':str(args.output)}))
    return code


if __name__=='__main__':raise SystemExit(main())
