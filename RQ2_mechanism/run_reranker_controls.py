"""Fair end-to-end controls: identical BGE, K=20, and direct NLI reranker.

All variants use cached atoms already distributed with the benchmark; no LLM is
called. Thus timings cover dense retrieval plus NLI verification, but explicitly
exclude atomization/inversion generation time.
"""
import argparse, ast, csv, json, os, platform, time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
from aligned_eval_data import load_aligned

INS = "Represent this sentence for searching relevant passages: "
K = 20

def parse(v):
    try: return ast.literal_eval(v) if v and v.strip() else []
    except (ValueError, SyntaxError): return []
def flat(x):
    return [z.strip() for y in (x or []) for z in (flat(y) if isinstance(y,list) else [y]) if isinstance(z,str) and z.strip()]
def inv(x):
    return [z.strip() for row in (x or []) for z in (row[:3] if isinstance(row,list) else []) if isinstance(z,str) and z.strip()]

def loader(path, kind):
    docs, seen, qs = [], {}, []
    def add(s):
        if s not in seen: seen[s]=len(docs); docs.append(s)
        return seen[s]
    with open(path,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if kind=='bill':
                q=r.get('Original_Text','').strip(); cs=[r.get('Conflict1 (Contradiction)',''),r.get('Conflict2 (Contradiction)','')]
                es=[r.get('Paraphrase_Structure_1 (Entailment)',''),r.get('Paraphrase_Structure_2 (Entailment)','')]
                gt={add(x) for x in cs if x}
                for x in es:
                    if x: add(x)
                atoms=inv(parse(r.get('Neg_Original_Text_atoms','')))
            else:
                q=r.get('query','').strip(); c=r.get('conflict','').strip(); gt={add(c)} if c else set(); atoms=inv(parse(r.get('negs_sourse_atoms','')))
                e=r.get('entail','').strip()
                if e: add(e)
            if q and gt: qs.append((q,gt,atoms))
    return docs,qs

def ndcg(order,gt,cut):
    dcg=sum(1/np.log2(i+2) for i,x in enumerate(order[:cut]) if x in gt)
    idcg=sum(1/np.log2(i+2) for i in range(min(len(gt),cut)))
    return dcg/idcg if idcg else 0

def run(name, docs, qs, embed, nli, device):
    d=embed.encode(docs,convert_to_tensor=True,normalize_embeddings=True,batch_size=64,show_progress_bar=True,device=device)
    ids={v:[] for v in ['original_bge','original_plus_nli','ticr_plus_nli']}; lat={v:[] for v in ['original_plus_nli','ticr_plus_nli']}
    # direct contradiction class is index 0 in this checkpoint (verified from config).
    for q,gt,atoms in tqdm(qs,desc=name):
        for variant in ['original_plus_nli','ticr_plus_nli']:
            probes=[q] if variant.startswith('original') else [f'{q} {a}' for a in atoms]
            if not probes: probes=[q]
            t=time.perf_counter()
            qemb=embed.encode([INS+x for x in probes],convert_to_tensor=True,normalize_embeddings=True,batch_size=64,show_progress_bar=False,device=device)
            score=torch.max(qemb@d.T,dim=0).values
            cand=torch.topk(score,K).indices.detach().cpu().tolist()
            if variant=='original_plus_nli': ids['original_bge'].append(cand)
            logits=nli.predict([[q,docs[i]] for i in cand],batch_size=20,show_progress_bar=False,convert_to_numpy=True)
            reranked=[x for _,x in sorted(zip(logits[:,0],cand),reverse=True)]
            ids[variant].append(reranked); torch.cuda.synchronize(); lat[variant].append((time.perf_counter()-t)*1000)
    out={}
    for v,ranks in ids.items():
        out[v]={f'hit@{k}':round(100*np.mean([any(x in gt for x in r[:k]) for r,(_,gt,_) in zip(ranks,qs)]),4) for k in [3,10,20]}
        out[v].update({f'ndcg@{k}':round(100*np.mean([ndcg(r,gt,k) for r,(_,gt,_) in zip(ranks,qs)]),4) for k in [3,10,20]})
        if v in lat: out[v]['mean_latency_ms_excl_generation']=round(float(np.mean(lat[v])),3)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='dataset'); ap.add_argument('--out',default='reranker_controls.json'); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    ep=os.environ['TICR_EMBED_PATH']; npth=os.environ['TICR_NLI_PATH']
    embed=SentenceTransformer(ep,device=a.device); nli=CrossEncoder(npth,device=a.device)
    allout={'protocol':{'encoder':'BAAI/bge-base-en-v1.5','reranker':'cross-encoder/nli-deberta-v3-base','candidate_k':K,'retrieval':'exact cosine','generation':'cached benchmark atoms; excluded from latency','gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'batch_embed':64,'batch_nli':20}}
    for name,file,kind in [('Bill-Contra','Bill-Contra.csv','bill'),('Juris-Logic','Juris-Logic.csv','juris')]:
        docs,records,audit=load_aligned(Path(a.data_dir)/file,kind)
        qs=[(x['q'],x['gt'],x['inverse_atoms']) for x in records]
        allout[name]={'queries':len(qs),'documents':len(docs),'data_audit':audit,'results':run(name,docs,qs,embed,nli,a.device)}
    json.dump(allout,open(a.out,'w'),indent=2); print(json.dumps(allout,indent=2))
if __name__=='__main__': main()
