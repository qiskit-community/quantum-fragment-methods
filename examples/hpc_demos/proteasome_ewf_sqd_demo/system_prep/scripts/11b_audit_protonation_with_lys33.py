#!/usr/bin/env python3
"""Stage 8b: audit protonation/formal-charge environment before adding H atoms.

Diagnostic only: no hydrogens, protonation states, water orientations, or
coordinates are changed.
"""
import argparse, csv, json
from pathlib import Path
import gemmi, numpy as np

ION = {"ASP":("deprotonated",-1),"GLU":("deprotonated",-1),"LYS":("protonated",1),
       "ARG":("protonated",1),"HIS":("manual_review",None),
       "CYS":("neutral_unless_evidence",0),"TYR":("neutral_unless_evidence",0)}
FOCUS={"ASP":["OD1","OD2"],"GLU":["OE1","OE2"],"LYS":["NZ"],
       "ARG":["NE","NH1","NH2"],"HIS":["ND1","NE2"],"CYS":["SG"],"TYR":["OH"]}
POLAR={"N","O","S"}; BB={"N","CA","C","O","OXT"}

def clean(x):
    x=str(x).strip(); return None if x in ("",".","?") else x

def asint(x):
    x=clean(x)
    try:return int(x) if x is not None else None
    except:return None

def parse(path):
    doc=gemmi.cif.read_file(str(path)); b=doc.sole_block()
    tags=["_atom_site.group_PDB","_atom_site.type_symbol","_atom_site.label_atom_id",
          "_atom_site.label_alt_id","_atom_site.label_comp_id","_atom_site.label_asym_id",
          "_atom_site.label_seq_id","_atom_site.auth_asym_id","_atom_site.auth_seq_id",
          "_atom_site.Cartn_x","_atom_site.Cartn_y","_atom_site.Cartn_z"]
    t=b.find(tags); poly={}; het={}
    for r in t:
        g,e,a,alt,c,lc,ls,ac,asq=[clean(r[i]) for i in range(9)]
        if alt not in (None,"A"): continue
        xyz=np.array([float(r[9]),float(r[10]),float(r[11])])
        if g=="ATOM" and lc is not None and asint(ls) is not None:
            k=(lc,asint(ls)); q=poly.setdefault(k,{"comp":c,"auth":asq,"atoms":{},"elem":{}})
            q["atoms"].setdefault(a,xyz); q["elem"].setdefault(a,e)
        elif g=="HETATM":
            k=(ac,asq,c); q=het.setdefault(k,{"comp":c,"atoms":{},"elem":{}})
            q["atoms"].setdefault(a,xyz); q["elem"].setdefault(a,e)
    return b,poly,het

def heavy(rec):
    return [x for n,x in rec["atoms"].items() if (rec["elem"].get(n) or "").upper()!="H"]

def md(a,b):
    if not a or not b:return None
    A=np.vstack(a); B=np.vstack(b); return float(np.sqrt((((A[:,None,:]-B[None,:,:])**2).sum(2)).min()))

def align(p3,p7,chain):
    P=[];Q=[]
    for k in sorted(set(p3)&set(p7)):
        if k[0]!=chain or p3[k]["comp"]!=p7[k]["comp"]:continue
        for a in ("N","CA","C","O"):
            if a in p3[k]["atoms"] and a in p7[k]["atoms"]:
                P.append(p3[k]["atoms"][a]);Q.append(p7[k]["atoms"][a])
    P=np.array(P);Q=np.array(Q); cp=P.mean(0);cq=Q.mean(0)
    U,S,Vt=np.linalg.svd((P-cp).T@(Q-cq));R=Vt.T@U.T
    if np.linalg.det(R)<0:Vt[-1]*=-1;R=Vt.T@U.T
    f=lambda x:R@(np.asarray(x)-cp)+cq
    rms=float(np.sqrt(np.mean(np.sum((np.vstack([f(x) for x in P])-Q)**2,axis=1))))
    return f,rms,len(P)

def ligand(het,comp,poly,chain):
    og=poly[(chain,1)]["atoms"]["OG1"]; C=[]
    for r in het.values():
        if r["comp"]!=comp:continue
        d=np.linalg.norm(r["atoms"]["B26"]-og) if "B26" in r["atoms"] else md(heavy(r),[og])
        C.append((d,r))
    if not C:raise RuntimeError(f"No {comp}")
    return sorted(C,key=lambda x:x[0])[0][1]

def comp_charge(block,comp):
    t=block.find(["_chem_comp_atom.comp_id","_chem_comp_atom.atom_id","_chem_comp_atom.charge"])
    if not t:return None
    vals=[]
    for r in t:
        if clean(r[0])!=comp:continue
        q=clean(r[2])
        try:vals.append(int(q))
        except:return None
    return sum(vals) if vals else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("cif3");ap.add_argument("cif7")
    ap.add_argument("--manifest",required=True);ap.add_argument("--out",required=True);ap.add_argument("--tsv",required=True)
    ap.add_argument("--chain",default="Y");ap.add_argument("--cutoff",type=float,default=4.0);a=ap.parse_args()
    man=json.loads(Path(a.manifest).read_text()); b3,p3,h3=parse(a.cif3);b7,p7,h7=parse(a.cif7)
    f,rms,nfit=align(p3,p7,a.chain); bo2=ligand(h3,"BO2",p3,a.chain); v8=ligand(h7,"6V8",p7,a.chain)
    Lb=[f(x) for x in heavy(bo2)]; Li=heavy(v8)
    kept=[(r["label_chain"],int(r["label_seq"])) for r in man["retained_residues"]]
    waters={}
    for w in man["retained_waters"]:
        c,s=w["source_water"].split(":"); rec=next(v for (cc,ss,cp),v in h7.items() if cc==c and str(ss)==s)
        waters[w["source_water"]]=heavy(rec)
    rows=[]
    for k in kept:
        rec=p7[k]; special=(k==(a.chain,1))
        if not special and rec["comp"] not in ION:continue
        names=["N","OG1"] if special else FOCUS[rec["comp"]]
        fc=[rec["atoms"][n] for n in names if n in rec["atoms"]]
        state,q=("SPECIAL_Thr1_boronate_coupled",None) if special else ION[rec["comp"]]
        polar=[]
        for j in kept:
            if j==k:continue
            rr=p7[j]
            for n,x in rr["atoms"].items():
                if (rr["elem"].get(n) or "").upper() in POLAR:
                    d=min(np.linalg.norm(x-y) for y in fc)
                    if d<=a.cutoff:polar.append((d,f"{j[0]}:{j[1]} {rr['comp']} {n}"))
        polar.sort()
        wd={wid:md(fc,wc) for wid,wc in waters.items()}
        rows.append({"residue":f"{k[0]}:{k[1]} {rec['comp']}","special_thr1":special,"hypothesis_only":state,
                     "default_charge":q,"focus_atoms":",".join(names),"to_6V8_A":md(fc,Li),"to_BO2_A":md(fc,Lb),
                     "nearest_water_A":min(wd.values()),"nearby_polar":"; ".join(f"{p}@{d:.3f}" for d,p in polar[:8])})
    ordinary=sum(r["default_charge"] for r in rows if r["default_charge"] is not None)
    report={"stage":"8b_protonation_audit","alignment_rmsd_A":round(rms,4),"alignment_atoms":nfit,
            "ordinary_default_charge_sum_excluding_Thr1_and_ligand":ordinary,
            "ionizable_groups":rows,"caps":{"ACE":"neutral","NME":"neutral"},
            "ligand_dictionary_charge":{"BO2":comp_charge(b3,"BO2"),"6V8":comp_charge(b7,"6V8")},
            "warnings":["Dictionary ligand charge is not automatically the covalent-adduct charge.",
                        "Thr1 N-terminus and boronate protonation must be chosen together.",
                        "No protonation state or hydrogen position is assigned by this script."],
            "next_step":"Choose explicit Thr1/boronate microstate(s), then add hydrogens and orient retained waters."}
    Path(a.out).write_text(json.dumps(report,indent=2)+"\n")
    with open(a.tsv,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=rows[0].keys(),delimiter="\t");w.writeheader();w.writerows(rows)
    print(f"Wrote {a.out}\nWrote {a.tsv}\n")
    print("Stage-8b protonation/formal-charge audit")
    print(f"  alignment RMSD: {rms:.3f} A over {nfit} backbone atoms")
    print(f"  ordinary default charge sum (excluding Thr1 + ligand): {ordinary:+d}\n")
    for r in rows:
        q='SPECIAL' if r['default_charge'] is None else f"{r['default_charge']:+d}"
        print(f"  {r['residue']:12s} q={q:7s} 6V8={r['to_6V8_A']:.3f} BO2={r['to_BO2_A']:.3f} water={r['nearest_water_A']:.3f}")
    print("\nLigand dictionary charge (metadata only; not final adduct charge)")
    for k,v in report['ligand_dictionary_charge'].items():print(f"  {k}: {v}")
    print("\nIMPORTANT: no final protonation state has been assigned.")
    print("Thr1 + boronate chemistry is the next explicit modeling decision.")

if __name__=='__main__':main()
