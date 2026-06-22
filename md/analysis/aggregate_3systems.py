#!/usr/bin/env python3
"""3-system MMAE-core self-RMSD comparison: vedotin (authentic clinical control)
vs cand_4 vs cand_5, N=3 each. Same metric/method throughout. Applies the
pre-registered reading and produces the comparative plot for §20.12."""
import math, statistics as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ANA='/home/galeito/ADCpedia/md/analysis'; REPS=['prod','prod_rep2','prod_rep3']; HALF=25.0
SYS={'cand_vedotin':('vedotin (clinical)','tab:green'),
     'cand_4':('cand_4','tab:blue'), 'cand_5':('cand_5','tab:orange')}

def series(cand,rep):
    xs=[];ys=[]
    for l in open(f'{ANA}/{cand}/mmae_selfrmsd_{rep}.xvg'):
        if l[0] in '#@':continue
        p=l.split(); xs.append(float(p[0])); ys.append(float(p[1])*10)
    return xs,ys
def repmean(cand,rep):
    xs,ys=series(cand,rep); return st.mean([v for t,v in zip(xs,ys) if t>=HALF])

data={c:[repmean(c,r) for r in REPS] for c in SYS}
print('=== MMAE-core self-RMSD, mean 25-50 ns (A) ===')
stats={}
for c,(lab,_) in SYS.items():
    v=data[c]; m=st.mean(v); sd=st.stdev(v); stats[c]=(m,sd,v)
    print(f'  {lab:18s}: reps={[round(x,2) for x in v]}  mean={m:.2f}  SD={sd:.2f}  range[{m-sd:.2f},{m+sd:.2f}]')

def welch(a,b):
    ma,mb=st.mean(a),st.mean(b); sa,sb=st.stdev(a),st.stdev(b); na=nb=3
    se=math.sqrt(sa*sa/na+sb*sb/nb); t=(ma-mb)/se if se else float('inf')
    df=(sa*sa/na+sb*sb/nb)**2/((sa*sa/na)**2/(na-1)+(sb*sb/nb)**2/(nb-1))
    try:
        from scipy import stats as S; p=2*S.t.sf(abs(t),df)
    except Exception: p=float('nan')
    return t,df,p
print('\n=== Welch t-tests vs vedotin ===')
for c in ['cand_4','cand_5']:
    t,df,p=welch(data['cand_vedotin'],data[c])
    print(f'  vedotin vs {c}: t={t:.2f} df={df:.2f} p={p:.3f}  -> {"separable" if p<0.05 else "NOT separable (overlap)"}')

# pre-registered reading
mv,sdv,_=stats['cand_vedotin']
lo_c=min(stats['cand_4'][0]-stats['cand_4'][1], stats['cand_5'][0]-stats['cand_5'][1])
hi_c=max(stats['cand_4'][0]+stats['cand_4'][1], stats['cand_5'][0]+stats['cand_5'][1])
in_range = (1.5 <= mv <= 5.0) and (mv+sdv > lo_c) and (mv-sdv < hi_c)
tight_low = (mv < 1.5 and sdv < 0.5)
print('\n=== PRE-REGISTERED READING ===')
if tight_low: verdict='CLINICAL LINKER STABILISES BETTER (vedotin low+tight <1.5 A)'
elif in_range: verdict='ANCHOR CONFIRMED (vedotin in candidate regime; self-RMSD does not separate the approved linker from generated ones)'
else: verdict='OUTLIER (vedotin far/erratic -> suspect build artifact)'
print(f'  vedotin {mv:.2f}+-{sdv:.2f} A vs candidate envelope [{lo_c:.2f},{hi_c:.2f}] -> {verdict}')

# plot
fig,(axL,axR)=plt.subplots(1,2,figsize=(13,5))
for c,(lab,col) in SYS.items():
    for i,r in enumerate(REPS):
        xs,ys=series(c,r); axL.plot(xs,ys,color=col,alpha=0.45,lw=0.7,label=lab if i==0 else None)
axL.axvline(HALF,ls=':',c='k',lw=0.8); axL.set_xlabel('time (ns)'); axL.set_ylabel('MMAE-core self-RMSD (A)')
axL.set_title('All replicate trajectories (N=3 each)'); axL.legend(fontsize=8)
order=['cand_4','cand_vedotin','cand_5']
for x,c in enumerate(order):
    m,sd,v=stats[c]; col=SYS[c][1]
    axR.bar(x,m,yerr=sd,capsize=8,color=col,alpha=0.6,width=0.55)
    for val in v: axR.plot(x,val,'o',color='k',ms=7,zorder=3)
axR.set_xticks(range(3)); axR.set_xticklabels([SYS[c][0] for c in order],fontsize=9)
axR.set_ylabel('self-RMSD mean 25-50 ns (A)'); axR.set_ylim(0,5.5)
axR.set_title('mean+-SD (N=3)\nVERDICT: '+('ANCHOR CONFIRMED' if in_range else verdict.split('(')[0]))
plt.tight_layout(); out=f'{ANA}/selfrmsd_3systems.png'; plt.savefig(out,dpi=130)
print(f'\nplot -> {out}')
