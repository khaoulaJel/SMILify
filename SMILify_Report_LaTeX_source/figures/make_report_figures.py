"""Report figures generated from traced result data (sources in REPORT_CLAIM_AUDIT.md).
Run: /home/nao48500/miniforge3/envs/pytorch3d/bin/python make_report_figures.py  (from SMILify_Report_LaTeX_source/)"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
for f in ['Regular','Bold','Italic']:
    fm.fontManager.addfont(f'/usr/share/fonts/google-carlito-fonts/Carlito-{f}.ttf')
plt.rcParams.update({'font.family':'Carlito','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
ACC='#16506B'; ORG='#C9792A'; GRN='#3C8D5A'; RED='#A8392E'; GRY='#9AA3A8'

# ---- 1. JAB: contrasts (forest) + stage progression ---------------------------------------
arms=[('B  no part-wise coarse-to-fine fitting',8.2,3.1,13.8),('K  pose held fixed during surface fitting',3.8,-0.7,7.1),
      ('J  no deformation / normal regularisation',5.0,-0.3,9.8),('D  no joint limits',4.0,0.8,7.1),
      ('I  learned dense correspondence',2.3,-11.2,11.9),('L  separate distal-leg groups',0.9,-1.2,4.8),
      ('H  learned pose initialisation',0.7,-1.2,15.2),('C  no part partition',0.5,-2.6,3.9),
      ('F  no scale regularisation',0.5,-0.3,1.4),('E  extra anterior joint limits',1.3,0.1,2.5),('G  allometric prior',0.0,-1.3,1.8)]
equiv={'F','E','G'}
fig,ax=plt.subplots(1,2,figsize=(7.6,3.7),gridspec_kw={'width_ratios':[1.9,1]})
a=ax[0]; a.axvspan(-2.5,2.5,color=GRN,alpha=0.12,lw=0); a.axvline(0,color='k',lw=0.8)
for i,(n,m,lo,hi) in enumerate(arms):
    y=len(arms)-1-i; k=n[0]
    c=GRN if k in equiv else (ORG if (lo>0 or k in 'JK') else GRY)
    a.plot([lo,hi],[y,y],color=c,lw=2); a.plot(m,y,'o',color=c,ms=6)
a.set_yticks(range(len(arms))); a.set_yticklabels([x[0] for x in arms][::-1],fontsize=10)
a.set_xlabel('Change in joint error vs production recipe\n(paired shift, % of Weber\'s length; right = worse)')
a.text(0,len(arms)-0.2,'equivalence margin ±2.5%',ha='center',fontsize=9,color=GRN)
a.set_xlim(-12,16)
b=ax[1]; st=[('leg-chain\nfitting',-8.4,-11.6,-5.4),('joint\nrefinement',-0.8,-2.1,0.8),('surface\ncoarse',-3.0,-6.4,1.0),('surface\nfine',-0.3,-1.2,0.1)]
for i,(n,m,lo,hi) in enumerate(st):
    b.bar(i,m,color=ACC if i==0 else '#7FA3B5',width=0.6); b.plot([i,i],[lo,hi],color='k',lw=1.2)
b.axhline(0,color='k',lw=0.8); b.set_xticks(range(4)); b.set_xticklabels([s[0] for s in st],fontsize=9)
b.set_ylabel('Step change (% WL)'); b.set_title('Per-stage change',fontsize=11)
plt.tight_layout(); plt.savefig('figures/fig_jab_contrasts.png',dpi=300); plt.close()

# ---- 2. surface-vs-skeleton dissociation --------------------------------------------------
pts={'A':(1.00,1.00),'B':(1.19,1.23),'C':(0.97,0.97),'D':(0.96,1.05),'E':(0.99,1.02),'F':(1.00,1.01),
     'G':(1.00,0.99),'H':(1.00,1.07),'I':(2.97,1.08),'J':(0.45,1.25),'K':(2.18,1.18),'L':(1.01,1.06)}
fig,a=plt.subplots(figsize=(4.6,3.5))
a.axhline(1,color=GRY,lw=0.8); a.axvline(1,color=GRY,lw=0.8)
a.fill_between([0.4,1],0.95,1,color=GRN,alpha=0.10,lw=0)
a.text(0.43,0.955,'better surface\nand better skeleton:\nno strategy',fontsize=9,color=GRN,va='bottom')
for k,(x,y) in pts.items():
    c=ORG if k=='J' else (ACC if k=='A' else GRY)
    a.plot(x,y,'o',color=c,ms=7); a.annotate(k,(x,y),xytext=(5,4),textcoords='offset points',fontsize=10)
a.set_xscale('log'); a.set_xlim(0.4,3.4); a.set_ylim(0.95,1.3)
a.set_xticks([0.5,1,2,3]); a.set_xticklabels(['0.5','1','2','3'])
a.set_xlabel('Surface distance (Chamfer) relative to recipe  (lower = closer surface)')
a.set_ylabel('Joint-level error ratio\n(higher = worse skeleton)')
plt.tight_layout(); plt.savefig('figures/fig_dissociation.png',dpi=300); plt.close()

# ---- 3. initialisation sensitivity --------------------------------------------------------
seg=['coxa','trochanter','femur','tibia','tarsus','pretarsus']
zero=[0.4204,0.1523,0.1588,0.1677,0.1734,0.1717]; gt=[0.4114,0.0866,0.0504,0.0217,0.0285,0.0140]
fig,ax=plt.subplots(1,2,figsize=(7.6,3.3),gridspec_kw={'width_ratios':[1.6,1]})
a=ax[0]; x=np.arange(6); w=0.38
a.bar(x-w/2,zero,w,color=GRY,label='default initialisation'); a.bar(x+w/2,gt,w,color=ACC,label='ground-truth initialisation')
a.axhline(0.0908,color=RED,ls='--',lw=1); a.text(5.45,0.098,'label-ambiguity floor (adjacent coxae)',ha='right',fontsize=8.5,color=RED)
a.set_xticks(x); a.set_xticklabels(seg,fontsize=9.5,rotation=15); a.set_ylabel('Leg-segment correspondence error')
a.legend(frameon=False,fontsize=9,loc='upper right'); a.set_title('Effect of exact initialisation per segment',fontsize=11)
b=ax[1]; names=['error at\nproximal joints','unstructured','error at\ndistal joints']; v=[0.768,0.883,0.939]
b.bar(range(3),v,color=[RED,GRY,GRN],width=0.6)
for i,t in enumerate(v): b.text(i,t+0.01,f'{t:.2f}',ha='center',fontsize=10)
b.set_ylim(0.6,1.0); b.set_xticks(range(3)); b.set_xticklabels(names,fontsize=9); b.set_ylabel('Part-assignment accuracy (legs)')
b.set_title('Same total error (≈23°)',fontsize=11)
plt.tight_layout(); plt.savefig('figures/fig_init_sensitivity.png',dpi=300); plt.close()

# ---- 4. local retrieval vs dense deployment ----------------------------------------------
fig,ax=plt.subplots(1,3,figsize=(7.6,3.0),gridspec_kw={'width_ratios':[1.5,0.8,0.8]})
a=ax[0]; s4=['coxa','trochanter','femur','tibia']; rig=[0.2689,0.0753,0.0881,0.0917]; lrn=[0.2012,0.0798,0.0969,0.1148]
x=np.arange(4); a.bar(x-0.2,rig,0.4,color=GRY,label='Euclidean nearest neighbour'); a.bar(x+0.2,lrn,0.4,color=ACC,label='learned descriptor')
a.set_xticks(x); a.set_xticklabels(s4,fontsize=9); a.set_ylabel('Within-part matching error'); a.legend(frameon=False,fontsize=8.5)
a.set_title('(a) local retrieval',fontsize=10.5)
b=ax[1]; b.bar([0,1],[28.1,1.8],color=[GRY,ACC],width=0.6); b.set_xticks([0,1]); b.set_xticklabels(['template-embedding\nlookup','learned\ndescriptor'],fontsize=8.5)
b.set_ylabel('Coxal template vertices\nreceiving a match (%)'); b.set_title('(b) coverage',fontsize=10.5)
for i,t in enumerate([28.1,1.8]): b.text(i,t+0.8,f'{t}',ha='center',fontsize=9.5)
c=ax[2]; c.bar([0,1],[0.0239,0.0586],color=[GRY,ACC],width=0.6); c.set_xticks([0,1]); c.set_xticklabels(['template-embedding\nlookup','learned\ndescriptor'],fontsize=8.5)
c.set_ylabel('Dense coxal error'); c.set_title('(c) deployed error',fontsize=10.5)
plt.tight_layout(); plt.savefig('figures/fig_local_vs_dense.png',dpi=300); plt.close()

# ---- 5. oracle ladder ---------------------------------------------------------------------
fig,a=plt.subplots(figsize=(5.8,3.0)); labels=['default','oracle\npart labels','oracle dense\ncorrespondence','upper bound\n(exact mesh)']
leg=[0.867,0.937,0.936,0.991]; sg=[0.831,0.860,0.874,0.974]; x=np.arange(4)
a.bar(x-0.2,leg,0.4,color=ACC,label='leg-level'); a.bar(x+0.2,sg,0.4,color=ORG,label='segment-level')
a.set_xticks(x); a.set_xticklabels(labels,fontsize=10); a.set_ylim(0.7,1.02); a.set_ylabel('Part-assignment accuracy')
for i in range(4):
    a.text(i-0.2,leg[i]+0.004,f'{leg[i]:.2f}',ha='center',fontsize=8.5); a.text(i+0.2,sg[i]+0.004,f'{sg[i]:.2f}',ha='center',fontsize=8.5)
a.legend(frameon=False,fontsize=9,loc='lower right'); plt.tight_layout(); plt.savefig('figures/fig_oracle_ladder.png',dpi=300); plt.close()

# ---- 6. corpus gallery ---------------------------------------------------------------------
names=[('atta-vollenweideri','Atta vollenweideri'),('diacamma-indicum','Diacamma indicum'),('ectatomma-tuberculatum','Ectatomma tuberculatum'),
       ('formica-japonica','Formica japonica'),('harpegnathos','Harpegnathos'),('leptogenys-wheeleri','Leptogenys wheeleri'),
       ('paraponera-clavata','Paraponera clavata'),('polyrhachis-bihamata','Polyrhachis bihamata')]
fig,axs=plt.subplots(2,4,figsize=(7.6,3.6)); fig.patch.set_facecolor('#0E2A3A')
for a,(f,t) in zip(axs.ravel(),names):
    im=Image.open(f'/w0/tmp/nao48500/allclean/prev/lateral/{f}.png').convert('RGBA'); bb=im.getbbox(); im=im.crop(bb)
    a.imshow(im); a.axis('off'); a.set_title(t,fontsize=9.5,color='white',style='italic')
plt.tight_layout(pad=0.6); plt.savefig('figures/fig_corpus_gallery.png',dpi=300,facecolor=fig.get_facecolor()); plt.close()
print('ok')
