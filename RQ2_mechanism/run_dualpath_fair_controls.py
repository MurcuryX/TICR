"""Pre-registered fair control for TICR candidate generation.

Original and TICR candidate pools both use the *same* frozen BGE encoder,
exact Top-20 depth, and the same fixed dual-path NLI score.  The only varied
component is candidate construction.  Cached benchmark inversions are used for
both arms, so this measures retrieval coverage rather than LLM generation.
"""
import argparse, ast, csv, json, os, time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
from aligned_eval_data import load_aligned

INS="Represent this sentence for searching relevant passages: "; K=20; TOPR=3
W_VEC=.9; W_LOGIC=.1; W_FULL=.5; W_INV=.5
def parse(v):
    try:return ast.literal_eval(v) if v and v.strip() else []
    except (ValueError,SyntaxError):return []
def inv(x): return [z.strip() for y in (x or []) for z in (y[:3] if isinstance(y,list) else []) if isinstance(z,str) and z.strip()]
def load(path,kind):
    docs=[]; seen={}; qs=[]
    def add(x):
        if x not in seen: seen[x]=len(docs);docs.append(x)
        return seen[x]
    with open(path,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if kind=='bill':
                q=r.get('Original_Text','').strip(); gt={add(x) for x in [r.get('Conflict1 (Contradiction)',''),r.get('Conflict2 (Contradiction)','')] if x}
                for x in [r.get('Paraphrase_Structure_1 (Entailment)',''),r.get('Paraphrase_Structure_2 (Entailment)','')]:
                    if x:add(x)
                a=inv(parse(r.get('Neg_Original_Text_atoms','')))
            else:
                q=r.get('query','').strip(); c=r.get('conflict','').strip(); gt={add(c)} if c else set(); e=r.get('entail','').strip()
                if e:add(e)
                a=inv(parse(r.get('negs_sourse_atoms','')))
            if q and gt and a:qs.append((q,gt,a))
    return docs,qs
def ndcg(rank,gt,k):
    dcg=sum(1/np.log2(i+2) for i,x in enumerate(rank[:k]) if x in gt); idcg=sum(1/np.log2(i+2) for i in range(min(k,len(gt))))
    return dcg/idcg if idcg else 0
def rerank(q,inv_atoms,cand,docs,origin_scores,model):
    direct=model.predict([[q,docs[i]] for i in cand],batch_size=20,show_progress_bar=False,convert_to_numpy=True)[:,0]
    pairs=[(i,a) for i in cand for a in inv_atoms]
    entail=model.predict([[docs[i],a] for i,a in pairs],batch_size=128,show_progress_bar=False,convert_to_numpy=True)[:,1]
    inv=np.array([np.mean(np.sort(entail[j*len(inv_atoms):(j+1)*len(inv_atoms)])[-TOPR:]) for j in range(len(cand))])
    raw=W_FULL*direct+W_INV*inv; z=(raw-raw.mean())/(raw.std()+1e-9)
    final=W_VEC*np.array(origin_scores)+W_LOGIC*z
    return [x for _,x in sorted(zip(final,cand),reverse=True)]
def run(name,docs,qs,e,n,device):
    d=e.encode(docs,convert_to_tensor=True,normalize_embeddings=True,batch_size=64,show_progress_bar=True,device=device)
    ranks={x:[] for x in ['original_dualpath','ticr_dualpath']}; lat={x:[] for x in ranks}
    for q,gt,a in tqdm(qs,desc=name):
        q0=e.encode(INS+q,convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False,device=device)
        p=e.encode([INS+q]+[INS+q+' '+x for x in a],convert_to_tensor=True,normalize_embeddings=True,batch_size=64,show_progress_bar=False,device=device)
        for key,score in [('original_dualpath',q0@d.T),('ticr_dualpath',torch.max(p[1:]@d.T,dim=0).values)]:
            torch.cuda.synchronize();t=time.perf_counter(); cand=torch.topk(score,K).indices.detach().cpu().tolist(); origin=(q0@d[cand].T).detach().cpu().numpy()
            ranks[key].append(rerank(q,a,cand,docs,origin,n));torch.cuda.synchronize();lat[key].append((time.perf_counter()-t)*1000)
    out={}
    for key,rr in ranks.items():
        out[key]={f'hit@{k}':round(100*np.mean([any(x in gt for x in r[:k]) for r,(_,gt,_) in zip(rr,qs)]),4) for k in [3,10,20]}
        out[key].update({f'ndcg@{k}':round(100*np.mean([ndcg(r,gt,k) for r,(_,gt,_) in zip(rr,qs)]),4) for k in [3,10,20]});out[key]['mean_latency_ms_excl_generation']=round(float(np.mean(lat[key])),3)
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='dataset');ap.add_argument('--out',default='dualpath_fair_controls.json');ap.add_argument('--device',default='cuda');a=ap.parse_args()
    e=SentenceTransformer(os.environ['TICR_EMBED_PATH'],device=a.device);n=CrossEncoder(os.environ['TICR_NLI_PATH'],device=a.device)
    out={'protocol':{'candidate_k':K,'reranker':'fixed dual path: .9 original-BGE + .1 zscore(.5 contradiction + .5 top-3 inverse entailment)','inversions':'cached benchmark atoms for both arms','gpu':torch.cuda.get_device_name(0),'batch_nli_inverse':128}}
    for name,fn,kind in [('Bill-Contra','Bill-Contra.csv','bill'),('Juris-Logic','Juris-Logic.csv','juris')]:
        docs,records,audit=load_aligned(Path(a.data_dir)/fn,kind)
        qs=[(x['q'],x['gt'],x['inverse_atoms']) for x in records]
        out[name]={'queries':len(qs),'documents':len(docs),'data_audit':audit,'results':run(name,docs,qs,e,n,a.device)}
    json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
