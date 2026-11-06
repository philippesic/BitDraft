from pathlib import Path
import collections, datetime, json, os, stat

P=Path('/home/philip/binary-eagle-decoding')
OUT=P/'runs/storage-purge-20261104-01'
C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
A=P/'checkouts/a8-qat-run-583480c7/runs/qat-a8-comparison-20261003-01'
R=P/'checkouts/recurrent-gpu-capture-20260928/runs/recurrent-full96-20260928'
D=P/'runs/dspark-a8-full-pool-20261006-01'
L=P/'runs/luna-continuous-a8-a1-native-order-20260930'
ROOTS=[A,R,D,L,C]
protected=set(json.loads((OUT/'protected-paths.json').read_text()))
retained_roots=[A/'arms/reference/checkpoints/step-000000043203-e000000-r000000043287',A/'arms/candidate/checkpoints/step-000000034731-e000000-r000000034797',L/'stages/gate-a8',L/'stages/gate-a1',C/'capture05']
retained_files=[
 D/'training-memory-r1-lane/training/checkpoints/step-000000034403-e000000-b000000068806-direct-t0000014400266826.pt',
 D/'training-memory-r1-lane/training/checkpoints/step-000000034879-e000000-b000000069758-direct-t0000014645059307.pt',
 D/'training-memory-r1-lane/attempts/562fba054c774917aff8797f56f6ed8b/evaluations/0183e50108a144a68d3c8024958bf337/export/trained.gguf',
 C/'training-checkpoints-control/70be756dd844c8152e689fe11d6790d4b54ff086cb61bb93a477013b3d1e0540/checkpoint-00014400.pt',
 C/'training-checkpoints-dpal/d244ca90287bc3e31a2ceb1d27d3663dfb8b16002c9eaeaad4ce7698ff67230f/checkpoint-00008629.pt',
 C/'fresh-training-checkpoints01/2f5ab40fcb81dddb10a938caa3498b02b9e33048d857124dde1965b9dcf670a4/checkpoint-00014400.pt',
 C/'main-controller02/jobs/0a18e91556697adace084a4010424fcac8eb5b39a1deeebdfc50bdc074c704ce/attempt-002/model.gguf',
 C/'valid-prestop-dpal01/zero-export/model.gguf',
 C/'fresh-training01/jobs/c8666ed82c18c01c2bea3c84b7d5be5ef817c6e619440d63eb9def34170e48a6/attempt-006/model.gguf',
 C/'fresh-training01/jobs/c8666ed82c18c01c2bea3c84b7d5be5ef817c6e619440d63eb9def34170e48a6/attempt-001/model.gguf',
 P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf',
 Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-added-replay-pair-20261103-01/true5760/model.gguf'),
 Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-added-replay-pair-20261103-01/true5760/checkpoint-00005760.pt'),
]
for path in retained_roots+retained_files:assert path.exists(),path
protected.update(str(x) for x in retained_files)
def identity(path):
    s=path.lstat()
    return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,mode=stat.S_IFMT(s.st_mode),uid=s.st_uid,allocated=s.st_blocks*512,nlink=s.st_nlink)
def walk(root):
    assert root.resolve()==root and root.is_dir(),root
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in files:yield Path(directory)/name
selected={};skipped=[];aliases=[]
def add(path,category):
    if str(path) in protected or any(path.is_relative_to(x) for x in retained_roots):
        skipped.append(str(path));return
    s=identity(path)
    assert s['mode']==stat.S_IFREG and s['uid']==os.getuid() and path.resolve()==path,path
    assert any(path.is_relative_to(root) for root in ROOTS)
    assert path.suffix not in ('.gguf','.json','.md','.py','.cpp','.cu','.so','.toml'),path
    selected[str(path)]={'path':str(path),'category':category,'identity':s}
for path in walk(A/'evidence-archive/checkpoints'):
    if not path.is_symlink() and path.suffix in ('.pt','.npz'):add(path,'old_A8_intermediate_checkpoints')
for path in walk(A):
    if not path.is_symlink() and 'development' in path.relative_to(A).parts and path.suffix=='.f32':add(path,'old_A8_evaluation_tensors')
for path in walk(R):
    if not path.is_symlink() and path.suffix in ('.f32','.npy'):add(path,'preparation_only_recurrent_capture')
for path in walk(D):
    if not path.is_symlink() and path.suffix=='.f32':add(path,'old_DSpark_teacher_tensors')
for name in ['step-000000034411-e000000-b000000068822-direct-t0000014404146586.pt','step-000000034465-e000000-b000000068930-direct-t0000014454888907.pt']:
    add(D/'training-memory-r1-lane/training/checkpoints'/name,'old_DSpark_intermediate_checkpoints')
luna_files=['native/d_d/heads.target_features.jsonl','native/d_d/heads.jsonl','native/d_d/rounds.jsonl','native/d_d/forced-rounds.jsonl','labels/rows.jsonl','labels/feature_rows.jsonl','labels/anchors.jsonl']
for directory in (L/'stages').glob('capture-*'):
    assert directory.is_dir() and not directory.is_symlink()
    for name in luna_files:
        path=directory/name
        if path.is_file() and not path.is_symlink():add(path,'old_Luna_raw_teacher_rows')
c_checkpoints=[
 'fresh-qualification-checkpoints01/de897761840a6ceca3b07d25ac22576e073de14a938f1360a263d4142d190da5/checkpoint-00000008.pt',
 'fresh-training-checkpoints01/10a6c107f25aa0a4aac3cdb6f4ff2beca1fdb8b633a981b696b05c36026b7e9b/checkpoint-00012499.pt',
 'qualification-checkpoints04/033076f8e7bbcb6ce908b02e99e5ef66adab732425a86224897f716ada072a01/checkpoint-00000008.pt',
 'qualification-checkpoints04/a50eff457f4a2236427f38c166dc4d1e31ec07fc2a8f627a8f760ded94cee4de/checkpoint-00000008.pt',
 'training-checkpoints-control/0f4120c8b90e3c7d9ed837e1de47c9fdf52b10c4eb078d380cc489231f7bc602/checkpoint-00012477.pt',
 'training-checkpoints-dpal/2b924a59a12ad256725e7d3e6a7e71aefa28fc4edd5959d3405a35c9c8fc6ed7/checkpoint-00004800.pt',
 'training-checkpoints-dpal/7d2a2c725e9e614c1b25a1f098ba291ee33631fa07e7c4563af815c91f2d88cb/checkpoint-00009318.pt',
]
for name in c_checkpoints:add(C/name,'old_F16_broader_intermediate_checkpoints')
for name in ['data','teacher','fresh-qualification01','fresh-training01','main-controller02','narrow32400-native01','parent-native01','qualification04','valid-prestop-dpal01']:
    for path in walk(C/name):
        if not path.is_symlink() and path.suffix=='.f32':add(path,'old_F16_broader_evaluation_arrays')
for root in ROOTS:
    for path in walk(root):
        if path.is_symlink() and str(path.resolve(strict=False)) in selected:
            assert str(path) not in protected
            aliases.append(dict(path=str(path),target=os.readlink(path)))
inodes=collections.defaultdict(list)
for item in selected.values():inodes[item['identity']['dev'],item['identity']['ino']].append(item)
totals=collections.defaultdict(lambda:dict(files=0,logical_bytes=0,expected_reclaim_bytes=0))
outside=[]
for items in inodes.values():
    first=items[0];s=first['identity'];category=first['category']
    if len(items)<s['nlink']:outside.append(dict(path=first['path'],selected_links=len(items),nlink=s['nlink']))
    else:totals[category]['expected_reclaim_bytes']+=s['allocated']
for item in selected.values():
    totals[item['category']]['files']+=1;totals[item['category']]['logical_bytes']+=item['identity']['size']
keep={str(x):identity(x) for x in retained_files}
for root in retained_roots:
    for path in walk(root):
        if path.is_file() and not path.is_symlink():keep[str(path)]=identity(path)
manifest=dict(schema='explicit_authorized_obsolete_artifact_purge_manifest_v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),roots=[str(x) for x in ROOTS],files=sorted(selected.values(),key=lambda x:x['path']),aliases=aliases,protected_files=keep,excluded_current_dependency_files=skipped,category_totals=dict(totals),selected_inodes_with_links_retained_elsewhere=outside,scope='User-authorized deletion of obsolete historical payloads only; source, current endpoints, useful historical final/best states and compact records retained. Historical exact tensor replay will require regeneration.')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(files=len(selected),aliases=len(aliases),protected_retained_files=len(keep),excluded_current_dependencies=len(skipped),category_totals=dict(totals),outside_hardlinks=len(outside))))
