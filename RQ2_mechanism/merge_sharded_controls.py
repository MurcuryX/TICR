"""Merge sharded matched-control JSON files by query-count weighting."""
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pattern',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); paths=sorted(Path().glob(args.pattern))
    if not paths: raise SystemExit('no shard files')
    merged={'protocol':{}, 'shards':[str(p) for p in paths]}
    names=set()
    for p in paths: names.update(json.loads(p.read_text()).keys())
    names.discard('protocol')
    for name in sorted(names):
        vals=[json.loads(p.read_text())[name] for p in paths if name in json.loads(p.read_text())]
        total=sum(v.get('queries',0) for v in vals)
        out={'queries':total,'documents':vals[0].get('documents'),'data_audit':vals[0].get('data_audit'), 'results':{}}
        keys=set().union(*(v.get('results',{}).keys() for v in vals))
        for key in keys:
            metric_keys=set().union(*(v.get('results',{}).get(key,{}).keys() for v in vals))
            out['results'][key]={}
            for metric in metric_keys:
                nums=[(v.get('queries',0),v.get('results',{}).get(key,{}).get(metric)) for v in vals]
                nums=[x for x in nums if x[1] is not None]
                if nums:
                    if metric == 'total_candidate_probes':
                        out['results'][key][metric] = round(sum(x for _, x in nums), 4)
                    else:
                        out['results'][key][metric]=round(sum(n*x for n,x in nums)/sum(n for n,x in nums),4)
        merged[name]=out
    Path(args.out).write_text(json.dumps(merged,indent=2))

if __name__=='__main__': main()
