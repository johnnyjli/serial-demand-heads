import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, os
SP=str(paths.ANALYSIS)
sys.argv=[None, SP]
sys.path.insert(0,SP)
import pairs
P=json.load(open(f'{SP}/pairs_q10.json'))
ref={(r['base'],r['post'],r['cell']):r for r in P}
for job in [("Llama-3.2-3B","Llama-3.2-3B-Instruct","pointer/w0"),("OLMo-2-1124-7B","OLMo-2-1124-7B-SFT","binding/w1"),("Qwen2.5-7B","Qwen2.5-7B-Instruct","arith/w0")]:
    o=pairs.one(job); r=ref[job]
    print(os.environ.get('E2_STAT'), os.environ.get('E2_CLEAN'), job, 'mine', [round(x,3) for x in o['mean_all']], 'json', [round(x,3) for x in r['mean_all']], 'bw mine', [round(x,3) for x in o.get('mean_both_wrong',[0,0])], 'json', [round(x,3) for x in r.get('mean_both_wrong',[0,0])])
