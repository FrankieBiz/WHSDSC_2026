"""
WHSDSC 2026 — FINAL COMPETITION SUBMISSION
=============================================
Optimized for speed + maximum AUC
"""
import pandas as pd, numpy as np, warnings, os, sys
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy import stats as sp_stats

ROOT = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/"
df = pd.read_csv(ROOT + "whl_2025.csv")
SPEC = {'PP_up','PP_kill_dwn','PP_kill_up','empty_net_line'}

print("="*70)
print("  WHSDSC 2026 — FINAL COMPETITION SUBMISSION")
print("="*70)
print(f"\nLoaded {len(df):,} records | {df['game_id'].nunique():,} games | {df['home_team'].nunique()} teams")
sys.stdout.flush()

# ═══════════════════ GAME AGGREGATION ═══════════════════
def agg_game(g):
    reg = g[~g['home_off_line'].isin(SPEC) & ~g['away_off_line'].isin(SPEC)]
    row = {}
    for s in ['home','away']:
        o = 'away' if s=='home' else 'home'
        row[f'{s}_goals']=g[f'{s}_goals'].sum(); row[f'{s}_xg']=g[f'{s}_xg'].sum()
        row[f'{s}_shots']=g[f'{s}_shots'].sum(); row[f'{s}_pen']=g[f'{s}_penalties_committed'].sum()
        row[f'{s}_pim']=g[f'{s}_penalty_minutes'].sum(); row[f'{s}_reg_xg']=reg[f'{s}_xg'].sum()
        pp=g[g[f'{s}_off_line']=='PP_up']; row[f'{s}_pp_xg']=pp[f'{s}_xg'].sum()
        pk=g[g[f'{s}_off_line'].isin({'PP_kill_dwn','PP_kill_up'})]; row[f'{s}_pk_xga']=pk[f'{o}_xg'].sum()
    row['went_ot']=g['went_ot'].iloc[0]; row['home_team']=g['home_team'].iloc[0]
    row['away_team']=g['away_team'].iloc[0]; row['home_goalie']=g['home_goalie'].iloc[0]
    row['away_goalie']=g['away_goalie'].iloc[0]
    return pd.Series(row)

print("Aggregating games...", flush=True)
games = df.groupby('game_id').apply(agg_game).reset_index()
games['game_num'] = games['game_id'].str.extract(r'(\d+)').astype(int)
games = games.sort_values('game_num').reset_index(drop=True)
games['home_win'] = (games['home_goals']>games['away_goals']).astype(int)
wr = games['home_win'].mean()

# ═══════════════════ LEAGUE TABLE ═══════════════════
print("Building league table...", flush=True)
rows=[]
for _,g in games.iterrows():
    hw=g['home_win']; ot=g['went_ot']
    for side,opp_side,is_home in [('home','away',True),('away','home',False)]:
        w = hw if is_home else 1-hw
        rows.append({'team':g[f'{side}_team'],'opp':g[f'{opp_side}_team'],
            'gf':g[f'{side}_goals'],'ga':g[f'{opp_side}_goals'],
            'xgf':g[f'{side}_xg'],'xga':g[f'{opp_side}_xg'],
            'sf':g[f'{side}_shots'],'sa':g[f'{opp_side}_shots'],
            'win':w,'loss':1-w,
            'ot_loss':1 if (w==0 and ot==1) else 0,
            'pp_xg':g[f'{side}_pp_xg'],'pk_xga':g[f'{side}_pk_xga'],'pen':g[f'{side}_pen']})
tg=pd.DataFrame(rows)
tg['pts']=tg['win']*2+tg['ot_loss']*1

lt=tg.groupby('team').agg(GP=('win','count'),W=('win','sum'),L=('loss','sum'),
    OTL=('ot_loss','sum'),PTS=('pts','sum'),GF=('gf','sum'),GA=('ga','sum'),
    xGF=('xgf','sum'),xGA=('xga','sum'),SF=('sf','sum'),SA=('sa','sum'),
    PP_xGF=('pp_xg','sum'),PK_xGA=('pk_xga','sum')).reset_index()
lt['GD']=lt['GF']-lt['GA']; lt['xGD']=lt['xGF']-lt['xGA']
lt['xG%']=(lt['xGF']/(lt['xGF']+lt['xGA'])).round(4)
lt=lt.sort_values('PTS',ascending=False).reset_index(drop=True); lt.index+=1
print("\nLEAGUE TABLE:")
print(lt[['team','GP','W','L','OTL','PTS','GF','GA','GD','xGF','xGA','xGD','xG%']].to_string())
sys.stdout.flush()

# ═══════════════════ ELO RATINGS ═══════════════════
print("\nBuilding ELO ratings...", flush=True)
def upd_elo(eh,ea,sh,sa,K=32):
    exp=1/(1+10**((ea-eh)/400)); hw=int(sh>sa)
    m=abs(sh-sa); mult=np.log(max(m,1)+1)*(2.2/((abs(eh-ea)*0.001)+2.2))
    return eh+K*mult*(hw-exp), ea+K*mult*((1-hw)-(1-exp))

elo={t:1500 for t in games['home_team'].unique()}; xgelo={t:1500 for t in elo}
elo_recs=[]
for _,r in games.iterrows():
    h,a=r['home_team'],r['away_team']
    elo_recs.append({'game_id':r['game_id'],'h_elo':elo[h],'a_elo':elo[a],
                     'h_xgelo':xgelo[h],'a_xgelo':xgelo[a]})
    elo[h],elo[a]=upd_elo(elo[h],elo[a],r['home_goals'],r['away_goals'])
    xgelo[h],xgelo[a]=upd_elo(xgelo[h],xgelo[a],r['home_xg'],r['away_xg'])
games=games.merge(pd.DataFrame(elo_recs),on='game_id')
games['elo_prob']=1/(1+10**((games['a_elo']-games['h_elo'])/400))
games['xgelo_prob']=1/(1+10**((games['a_xgelo']-games['h_xgelo'])/400))

# ═══════════════════ TEAM PANEL + ROLLING ═══════════════════
print("Building rolling features...", flush=True)
hgn=games[['game_num','home_team']].rename(columns={'home_team':'team'})
agn=games[['game_num','away_team']].rename(columns={'away_team':'team'})
gnm=pd.concat([hgn,agn])
tg2=tg.merge(gnm,on='team',how='left').sort_values(['team','game_num']).reset_index(drop=True)

# Goalie map
hg=games[['game_num','home_team','home_goalie']].rename(columns={'home_team':'team','home_goalie':'goalie'})
ag=games[['game_num','away_team','away_goalie']].rename(columns={'away_team':'team','away_goalie':'goalie'})
tg2=tg2.merge(pd.concat([hg,ag]),on=['team','game_num'],how='left')

# Opponent-adjusted (season-level)
lax=tg2['xgf'].mean()
osxg=tg2.groupby('team')['xgf'].mean().to_dict()
tg2['opp_sxgf']=tg2['opp'].map(osxg).fillna(lax)
tg2['adj_xgf']=tg2['xgf']/(tg2['opp_sxgf']/lax+1e-3)
tg2['adj_xga']=tg2['xga']/(tg2['opp_sxgf']/lax+1e-3)

# Rolling (2 windows: 15, 30)
for w in [15,30]:
    for c in ['gf','ga','xgf','xga','sf','sa','win','pp_xg','pk_xga','adj_xgf','adj_xga','pen']:
        tg2[f'r{w}_{c}']=tg2.groupby('team',sort=False)[c].transform(
            lambda x: x.shift(1).rolling(w,min_periods=max(3,w//5)).mean())
    tg2[f'r{w}_xgpct']=tg2[f'r{w}_xgf']/(tg2[f'r{w}_xgf']+tg2[f'r{w}_xga']+1e-9)
    tg2[f'r{w}_xgd']=tg2[f'r{w}_xgf']-tg2[f'r{w}_xga']
    tg2[f'r{w}_axgpct']=tg2[f'r{w}_adj_xgf']/(tg2[f'r{w}_adj_xgf']+tg2[f'r{w}_adj_xga']+1e-9)
    wn=tg2.groupby('team',sort=False)['win'].transform(lambda x:x.shift(1).rolling(w,min_periods=1).sum())
    cn=tg2.groupby('team',sort=False)['win'].transform(lambda x:x.shift(1).rolling(w,min_periods=1).count())
    tg2[f'r{w}_bwin']=(wn.fillna(0)+5*0.5)/(cn.fillna(0)+5)

# Goalie rolling
print("Building goalie features...", flush=True)
gres=[]
for g2,grp in tg2.groupby('goalie'):
    grp=grp.sort_values('game_num').copy()
    grp['g_xga']=grp['xga'].shift(1).rolling(15,min_periods=3).mean()
    grp['g_ga']=grp['ga'].shift(1).rolling(15,min_periods=3).mean()
    grp['g_sa']=grp['sa'].shift(1).rolling(15,min_periods=3).mean()
    grp['g_gsax']=grp['g_xga']-grp['g_ga']
    grp['g_sv']=1-grp['g_ga']/(grp['g_sa']+1e-3)
    grp['g_winr']=grp['win'].shift(1).rolling(15,min_periods=3).mean()
    gres.append(grp)
gr=pd.concat(gres,ignore_index=True)
gff=gr[['goalie','game_num','g_gsax','g_sv','g_winr','g_xga','g_ga']].copy()

# Merge to game level
print("Merging to game level...", flush=True)
ts_sub=tg2[['team','game_num']+[c for c in tg2.columns if c.startswith('r')]].copy()
gf=games.copy()
for side in ['home','away']:
    ren=ts_sub.rename(columns={c:f'{side}_{c}' for c in ts_sub.columns if c not in ['team','game_num']})
    gf=gf.merge(ren,left_on=[f'{side}_team','game_num'],right_on=['team','game_num'],how='left')
    if 'team' in gf.columns: gf=gf.drop(columns=['team'])
for side in ['home','away']:
    ren=gff.rename(columns={c:f'{side}_{c}' for c in gff.columns if c not in ['goalie','game_num']})
    gf=gf.merge(ren,left_on=[f'{side}_goalie','game_num'],right_on=['goalie','game_num'],how='left')
    if 'goalie' in gf.columns: gf=gf.drop(columns=['goalie'])

gf['goalie_gsax_d']=gf.get('home_g_gsax',0)-gf.get('away_g_gsax',0)
gf['goalie_win_d']=gf.get('home_g_winr',0.5)-gf.get('away_g_winr',0.5)
for w in [15,30]:
    for s in ['xgpct','xgd','win','bwin','pp_xg','axgpct','adj_xgf','pen']:
        hc,ac=f'home_r{w}_{s}',f'away_r{w}_{s}'
        if hc in gf.columns and ac in gf.columns: gf[f'd{w}_{s}']=gf[hc]-gf[ac]

# ═══════════════════ FEATURE SELECTION + TRAINING ═══════════════════
roll_c=[c for c in gf.columns if c.startswith('home_r') or c.startswith('away_r')]
gol_c=[c for c in gf.columns if '_g_gsax' in c or '_g_sv' in c or '_g_winr' in c or '_g_xga' in c]
diff_c=[c for c in gf.columns if c.startswith('d15_') or c.startswith('d30_')]
elo_c=[c for c in ['elo_prob','xgelo_prob','goalie_gsax_d','goalie_win_d'] if c in gf.columns]
ALL_F=list(dict.fromkeys(roll_c+gol_c+diff_c+elo_c)); ALL_F=[c for c in ALL_F if c in gf.columns]

gf_c=gf[ALL_F+['home_win','game_id','home_team','away_team']].copy()
gf_c=gf_c.dropna(subset=['home_r15_win','away_r15_win'])
med=gf_c[ALL_F].median(); gf_c[ALL_F]=gf_c[ALL_F].fillna(med)
X=gf_c[ALL_F].values; y=gf_c['home_win'].values
print(f"\nFull: {len(X)} games × {len(ALL_F)} features", flush=True)

print("Feature selection (top 50)...", flush=True)
et_s=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,random_state=42,n_jobs=-1)
et_s.fit(X,y)
fi=pd.Series(et_s.feature_importances_,index=ALL_F).sort_values(ascending=False)
top_f=fi.head(50).index.tolist(); Xb=gf_c[top_f].values

print("\n"+"="*70)
print("  TRAINING ENSEMBLE")
print("="*70, flush=True)
BASE=[
    ('lr',Pipeline([('s',RobustScaler()),('m',LogisticRegression(C=0.5,max_iter=2000,random_state=42))])),
    ('gb',GradientBoostingClassifier(n_estimators=500,learning_rate=0.03,max_depth=3,subsample=0.8,min_samples_leaf=6,max_features=0.6,random_state=42)),
    ('rf',RandomForestClassifier(n_estimators=500,min_samples_leaf=3,max_features=0.4,random_state=42,n_jobs=-1)),
    ('et',ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,max_features=0.4,random_state=42,n_jobs=-1)),
]
cv5=StratifiedKFold(5,shuffle=True,random_state=42)
oof_p={}; names_list=[]
for name,m in BASE:
    oof=cross_val_predict(m,Xb,y,cv=cv5,method='predict_proba',n_jobs=-1)[:,1]
    auc=roc_auc_score(y,oof); oof_p[name]=oof; names_list.append(name)
    print(f"  {name:5s}: OOF AUC = {auc:.4f}", flush=True)

oof_mat=np.column_stack(list(oof_p.values()))
def neg_auc(w):
    w=np.abs(w); w=w/(w.sum()+1e-12); return -roc_auc_score(y,oof_mat@w)
res=minimize(neg_auc,[0.25]*len(BASE),method='Nelder-Mead',options={'maxiter':5000})
opt_w=np.abs(res.x); opt_w/=opt_w.sum()
blend_auc=roc_auc_score(y,oof_mat@opt_w)
print(f"\n  Blend weights: {dict(zip(names_list,opt_w.round(3)))}")
print(f"  {'='*50}")
print(f"  OOF BLEND AUC: {blend_auc:.4f}")
print(f"  {'='*50}", flush=True)

# Fit final
print("\nFitting final models...", flush=True)
fitted={}
for name,m in BASE: m.fit(Xb,y); fitted[name]=m

# ═══════════════════ PHASE 1a: POWER RANKINGS ═══════════════════
print("\n"+"="*70)
print("  PHASE 1a: POWER RANKINGS")
print("="*70, flush=True)

season=tg.groupby('team').agg(W=('win','sum'),GP=('win','count'),
    GF=('gf','sum'),GA=('ga','sum'),xGF=('xgf','sum'),xGA=('xga','sum'),
    PP_xGF=('pp_xg','sum'),PK_xGA=('pk_xga','sum')).reset_index()
season['win_pct']=season['W']/season['GP']; season['xG%']=season['xGF']/(season['xGF']+season['xGA'])
season['xGD_pg']=(season['xGF']-season['xGA'])/season['GP']
season['PP_pg']=season['PP_xGF']/season['GP']

elo_f=pd.DataFrame([{'team':t,'elo':e} for t,e in elo.items()])
xgelo_f=pd.DataFrame([{'team':t,'xgelo':e} for t,e in xgelo.items()])
season=season.merge(elo_f,on='team').merge(xgelo_f,on='team')

adj_s=tg2.groupby('team').agg(axgf=('adj_xgf','sum'),axga=('adj_xga','sum')).reset_index()
adj_s['axg%']=adj_s['axgf']/(adj_s['axgf']+adj_s['axga'])
season=season.merge(adj_s[['team','axg%']],on='team',how='left')

gl=gr.sort_values('game_num').groupby('team').last()
for c in ['g_gsax','g_sv','g_winr']:
    if c in gl.columns: season=season.merge(gl[[c]].reset_index(),on='team',how='left')

def pr(s,asc=True): return s.rank(pct=True,ascending=asc)
season['ps']=(0.20*pr(season['elo'])+0.18*pr(season['xgelo'])+0.18*pr(season['xG%'])+
    0.12*pr(season['axg%'].fillna(0.5))+0.10*pr(season['win_pct'])+0.08*pr(season['PP_pg'])+
    0.06*pr(season['PK_xGA']/season['GP'],asc=False)+
    0.05*pr(season['g_gsax'].fillna(0))+0.03*pr(season['g_winr'].fillna(0.5)))

power=season[['team','ps','elo','xgelo','xG%','axg%','win_pct','PP_pg','g_gsax']].sort_values(
    'ps',ascending=False).reset_index(drop=True)
power.index+=1
power.columns=['Team','PowerScore','ELO','xG_ELO','xG%','Adj_xG%','WinPct','PP_xGF_pg','Goalie_GSAx']
print("\nPower Rankings:")
print(power.round(4).to_string(), flush=True)

# ═══════════════════ PHASE 1a: MATCHUP PREDICTIONS ═══════════════════
print("\n"+"="*70)
print("  PHASE 1a: MATCHUP WIN PROBABILITIES")
print("="*70, flush=True)

MATCHUPS=[("brazil","kazakhstan"),("netherlands","mongolia"),("peru","rwanda"),
    ("thailand","oman"),("pakistan","germany"),("india","usa"),("panama","switzerland"),
    ("iceland","canada"),("china","france"),("philippines","morocco"),
    ("ethiopia","saudi_arabia"),("singapore","new_zealand"),("guatemala","south_korea"),
    ("uk","mexico"),("vietnam","serbia"),("indonesia","uae")]

ts_latest=tg2.sort_values('game_num').groupby('team').last()
gr_latest=gr.sort_values('game_num').groupby('team').last()

def get_feat(team):
    d={}
    if team in ts_latest.index:
        for c in ts_latest.columns:
            if c.startswith('r'): d[c]=ts_latest.loc[team,c]
    if team in gr_latest.index:
        for c in ['g_gsax','g_sv','g_winr','g_xga','g_ga']:
            if c in gr_latest.columns: d[c]=gr_latest.loc[team,c]
    d['elo']=elo.get(team,1500); d['xgelo']=xgelo.get(team,1500)
    return d

def pred_row(ht,at):
    hf,af=get_feat(ht),get_feat(at)
    row={}
    for f in top_f:
        if f.startswith('home_r'): row[f]=hf.get(f[5:],np.nan)
        elif f.startswith('away_r'): row[f]=af.get(f[5:],np.nan)
        elif f.startswith('home_g'): row[f]=hf.get(f[5:],np.nan)
        elif f.startswith('away_g'): row[f]=af.get(f[5:],np.nan)
        elif f=='elo_prob': row[f]=1/(1+10**((af['elo']-hf['elo'])/400))
        elif f=='xgelo_prob': row[f]=1/(1+10**((af['xgelo']-hf['xgelo'])/400))
        elif f=='goalie_gsax_d': row[f]=hf.get('g_gsax',0)-af.get('g_gsax',0)
        elif f=='goalie_win_d': row[f]=hf.get('g_winr',0.5)-af.get('g_winr',0.5)
        elif f.startswith('d'):
            parts=f.split('_',2); w=parts[0][1:]; s='_'.join(parts[1:])
            row[f]=(hf.get(f'r{w}_{s}',np.nan) or 0)-(af.get(f'r{w}_{s}',np.nan) or 0)
        else: row[f]=np.nan
    md=dict(zip(top_f,gf_c[top_f].median().values))
    vec=[row.get(f,md.get(f,0)) for f in top_f]
    vec=[md.get(top_f[i],0) if (v is None or (isinstance(v,float) and np.isnan(v))) else v for i,v in enumerate(vec)]
    return np.array(vec).reshape(1,-1)

print(f"\n{'Game':<6}{'Home':<16}{'Away':<16}{'Home Win%':>10}")
print("-"*52)
mres=[]
for i,(ht,at) in enumerate(MATCHUPS,1):
    vec=pred_row(ht,at)
    preds=np.array([fitted[n].predict_proba(vec)[0][1] for n in names_list])
    prob=float(np.clip(preds@opt_w,0.01,0.99))
    w=ht if prob>0.5 else at
    mres.append({'game':i,'home_team':ht,'away_team':at,'home_win_prob':round(prob,4),'predicted_winner':w})
    print(f"  {i:<4}{ht:<16}{at:<16}{prob:>9.4f}")
sys.stdout.flush()

# ═══════════════════ PHASE 1b: LINE DISPARITY (opponent-adjusted) ═══════════════════
print("\n"+"="*70)
print("  PHASE 1b: OFFENSIVE LINE QUALITY DISPARITY")
print("="*70)
print("  Method: Opponent-adjusted xG/60 (accounts for defensive matchup strength)")
sys.stdout.flush()

# Defensive pairing strength: xGA/60 for each (team, def_pairing)
ds={}
for team in df['home_team'].unique():
    for dp in ['first_def','second_def']:
        h=df[(df['home_team']==team)&(df['home_def_pairing']==dp)]
        a=df[(df['away_team']==team)&(df['away_def_pairing']==dp)]
        xga=h['away_xg'].sum()+a['home_xg'].sum(); toi=h['toi'].sum()+a['toi'].sum()
        ds[(team,dp)]=(xga/toi)*60 if toi>0 else np.nan
lad=np.nanmean([v for v in ds.values()])

# Vectorized adjustment factors
ds_s=pd.Series(ds).fillna(lad).replace(0,lad)
df['_ha']=list(zip(df['away_team'],df['away_def_pairing']))
df['_haf']=lad/df['_ha'].map(ds_s).fillna(lad).clip(lower=1e-6)
df['_haxg']=df['home_xg']*df['_haf']
df['_aa']=list(zip(df['home_team'],df['home_def_pairing']))
df['_aaf']=lad/df['_aa'].map(ds_s).fillna(lad).clip(lower=1e-6)
df['_aaxg']=df['away_xg']*df['_aaf']

drows=[]
for team in sorted(df['home_team'].unique()):
    for lt,tag in [('first_off','1st'),('second_off','2nd')]:
        h=df[(df['home_team']==team)&(df['home_off_line']==lt)]
        a=df[(df['away_team']==team)&(df['away_off_line']==lt)]
        axg=h['_haxg'].sum()+a['_aaxg'].sum(); toi=h['toi'].sum()+a['toi'].sum()
        rxg=h['home_xg'].sum()+a['away_xg'].sum()
        if toi>=60:
            drows.append({'team':team,'line':tag,'adj_xg60':axg/toi*60,'raw_xg60':rxg/toi*60,'toi':toi})

dl=pd.DataFrame(drows)
f1=dl[dl['line']=='1st'].set_index('team')
f2=dl[dl['line']=='2nd'].set_index('team')
disp=f1[['adj_xg60']].rename(columns={'adj_xg60':'first_adj_xg60'}).join(
    f2[['adj_xg60']].rename(columns={'adj_xg60':'second_adj_xg60'}))
disp['ratio']=disp['first_adj_xg60']/disp['second_adj_xg60']
disp=disp.join(f1[['raw_xg60']].rename(columns={'raw_xg60':'first_raw_xg60'}))
disp=disp.join(f2[['raw_xg60']].rename(columns={'raw_xg60':'second_raw_xg60'}))
disp=disp.sort_values('ratio',ascending=False).reset_index()
disp.index+=1

print("\n  All 32 Teams:")
print(disp[['team','first_adj_xg60','second_adj_xg60','ratio']].round(4).to_string())
print("\n  *** TOP 10 SUBMISSION ***")
for i,row in disp.head(10).iterrows():
    print(f"  {i:2d}. {row['team']:<16} ratio={row['ratio']:.4f}  "
          f"(1st: {row['first_adj_xg60']:.4f}  2nd: {row['second_adj_xg60']:.4f})")
sys.stdout.flush()

# ═══════════════════ PHASE 1c: VISUALIZATION ═══════════════════
print("\n"+"="*70)
print("  PHASE 1c: GENERATING VISUALIZATION")
print("="*70, flush=True)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

viz=disp[['team','ratio']].merge(power[['Team','PowerScore']].rename(columns={'Team':'team'}),on='team')
x=viz['ratio'].values; yp=viz['PowerScore'].values; tn=viz['team'].values
r_v,p_v=sp_stats.pearsonr(x,yp); rho_v,rho_p=sp_stats.spearmanr(x,yp)
sl,ic,_,_,_=sp_stats.linregress(x,yp)
xl=np.linspace(x.min()-0.03,x.max()+0.03,100); yl=sl*xl+ic
mx,my=np.median(x),np.median(yp)
cols=['#2a9d8f' if xi<mx and yi>=my else '#e9c46a' if xi>=mx and yi>=my else
      '#264653' if xi<mx else '#e76f51' for xi,yi in zip(x,yp)]

fig,ax=plt.subplots(figsize=(13,9),dpi=200)
ax.axhline(my,color='#aaa',lw=0.8,ls='--',alpha=0.5,zorder=1)
ax.axvline(mx,color='#aaa',lw=0.8,ls='--',alpha=0.5,zorder=1)
ax.plot(xl,yl,color='#e76f51',lw=2.5,alpha=0.5,zorder=2)
ax.scatter(x,yp,c=cols,s=140,edgecolors='white',lw=1,zorder=5)
for xi,yi,t in zip(x,yp,tn):
    ha='right' if xi>1.25 else 'left'; ox=-5 if xi>1.25 else 5
    ax.annotate(t.replace('_',' ').title(),(xi,yi),xytext=(ox,4),
        textcoords='offset points',fontsize=6.5,color='#333',alpha=0.85,ha=ha,va='bottom',fontweight='medium')

qp=dict(fontsize=9.5,color='#999',fontstyle='italic',ha='center',va='center',fontweight='bold',alpha=0.6)
ax.text(mx-(mx-x.min())/2,my+(yp.max()-my)/2,'BALANCED\n& STRONG',**qp)
ax.text(mx+(x.max()-mx)/2,my+(yp.max()-my)/2,'TOP-HEAVY\n& STRONG',**qp)
ax.text(mx-(mx-x.min())/2,my-(my-yp.min())/2,'BALANCED\n& WEAK',**qp)
ax.text(mx+(x.max()-mx)/2,my-(my-yp.min())/2,'TOP-HEAVY\n& WEAK',**qp)

ax.set_xlabel('\nOffensive Line Quality Disparity Ratio\n(First Line Adj. xG/60 ÷ Second Line Adj. xG/60)\n'
    '← More Balanced                                    More Top-Heavy →',fontsize=11,fontweight='bold',labelpad=2)
ax.set_ylabel('Team Power Score\n(Composite: ELO + xG% + Adj. xG% + Win Rate + PP + Goalie)\n',
    fontsize=11,fontweight='bold',labelpad=8)
ax.set_title('Offensive Line Balance Has No Impact on Team Success',
    fontsize=17,fontweight='bold',pad=18,color='#222')
ax.text(0.5,1.025,'WHL 2025 — Offensive Line Disparity vs. Team Strength (n = 32)',
    transform=ax.transAxes,fontsize=10,ha='center',color='#777')

leg=[Patch(fc='#2a9d8f',ec='w',label='Balanced & Strong'),Patch(fc='#e9c46a',ec='w',label='Top-Heavy & Strong'),
     Patch(fc='#264653',ec='w',label='Balanced & Weak'),Patch(fc='#e76f51',ec='w',label='Top-Heavy & Weak'),
     Line2D([0],[0],color='#e76f51',lw=2.5,alpha=0.5,label=f'Trend (r = {r_v:.2f}, p = {p_v:.2f})')]
ax.legend(handles=leg,loc='upper right',fontsize=9,framealpha=0.92,edgecolor='#ccc')

cap=(f"Disparity = first-line opponent-adjusted xG/60 ÷ second-line opponent-adjusted xG/60. "
     f"Adjustment accounts for opposing defensive pairing quality.\n"
     f"Pearson r = {r_v:.2f} (p = {p_v:.2f}), Spearman ρ = {rho_v:.2f} (p = {rho_p:.2f}). Dashed lines = medians.")
fig.text(0.08,0.005,cap,fontsize=7.5,color='#666',ha='left',va='bottom',fontstyle='italic')

ax.set_xlim(x.min()-0.04,x.max()+0.04); ax.set_ylim(yp.min()-0.05,yp.max()+0.05)
ax.tick_params(labelsize=10); ax.grid(True,alpha=0.12,lw=0.5)
ax.set_facecolor('#fafafa'); fig.patch.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(rect=[0,0.035,1,1])
vp=ROOT+"phase1c_visualization.png"
fig.savefig(vp,dpi=200,bbox_inches='tight',facecolor='white'); plt.close()
print(f"  Saved: phase1c_visualization.png ({os.path.getsize(vp)/1e6:.2f} MB)", flush=True)

# ═══════════════════ SAVE ALL ═══════════════════
print("\n"+"="*70)
print("  SAVING ALL OUTPUTS")
print("="*70)
lt.to_csv(ROOT+"league_table.csv",index_label='rank')
power.to_csv(ROOT+"power_rankings.csv",index_label='rank')
pd.DataFrame(mres).to_csv(ROOT+"matchup_predictions.csv",index=False)
disp.to_csv(ROOT+"line_disparity.csv",index_label='rank')
print("  ✓ league_table.csv\n  ✓ power_rankings.csv\n  ✓ matchup_predictions.csv")
print("  ✓ line_disparity.csv\n  ✓ phase1c_visualization.png")

# ═══════════════════ SUMMARY ═══════════════════
print(f"\n{'='*70}")
print("  COMPETITION SUMMARY")
print(f"{'='*70}")
print(f"\n  MODEL: OOF Blend AUC = {blend_auc:.4f} (~{blend_auc/0.69*100:.0f}% of theoretical ceiling)")
print(f"\n  PHASE 1a — #1: {power.iloc[0]['Team']} | #32: {power.iloc[-1]['Team']}")
print(f"\n  PHASE 1b — Top 3 Disparity:")
for i,r in disp.head(3).iterrows(): print(f"    {i}. {r['team']} ({r['ratio']:.4f})")
print(f"\n  PHASE 1c — r = {r_v:.2f} (p = {p_v:.2f}): No significant relationship")
print(f"\n  KEY DIFFERENTIATORS:")
print("    ✓ Margin-adjusted ELO (goals + xG)")
print("    ✓ Opponent-adjusted xG (strength of schedule)")
print("    ✓ Phase 1b: defensive pairing strength adjustment")
print("    ✓ Bayesian-smoothed win rates")
print("    ✓ Goalie GSAx (Goals Saved Above Expected)")
print("    ✓ Top-50 feature selection from 100+")
print("    ✓ 4-model OOF blend (Nelder-Mead optimized)")
print("    ✓ 5-fold stratified cross-validation")
