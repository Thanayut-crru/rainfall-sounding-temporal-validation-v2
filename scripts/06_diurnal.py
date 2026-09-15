"""Confirmed post-sounding 6-hour targets; fail closed on unknown timing."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

PERIODS = [('morning_07_13',7,13), ('afternoon_13_19',13,19), ('evening_19_01next',19,25)]
MODELS = ['LR','SVM','RF','XGB','MLP']


def aggregate_periods(intervals):
    x = intervals.copy()
    for c in ['forecast_date','interval_start','interval_end']:
        x[c] = pd.to_datetime(x[c])
    start = (x.interval_start-x.forecast_date).dt.total_seconds()/3600
    end = (x.interval_end-x.forecast_date).dt.total_seconds()/3600
    parts=[]
    for period,lo,hi in PERIODS:
        q=x[(start>=lo)&(end<=hi)]
        a=q.groupby(['forecast_date','station']).rain_mm.agg(['sum','count','size']).reset_index()
        a.loc[(a['count']!=2)|(a['size']!=2),'sum']=np.nan
        a['period']=period
        parts.append(a.rename(columns={'forecast_date':'date','sum':'rain_mm'}))
    return pd.concat(parts,ignore_index=True)


def main():
    from model_common import fit_predict
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=Path(os.environ.get('RAINFALL_PROJECT_DIR',Path(__file__).resolve().parent.parent))
    out=Path(os.environ.get('RAINFALL_OUTPUT_DIR',base/'outputs'))
    figdir=out/'figures'; figdir.mkdir(parents=True,exist_ok=True)
    path=out/'temporal_alignment_status.json'
    timing=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    columns=['station','period','model','feature_set','n_train','n_test','rain_rate_test','roc_auc','pr_auc','brier']
    if timing.get('three_hour_status')!='confirmed':
        pd.DataFrame(columns=columns).to_csv(out/'diurnal_results.csv',index=False)
        (out/'diurnal_status.json').write_text(json.dumps({'status':'pending_metadata','reason':'Provider-confirmed interval-end/date/timezone metadata required. No forecast skill calculated.'},indent=2),encoding='utf-8')
        print('DIURNAL PENDING: no forecast claim or provisional interval AUC generated.')
        return
    intervals=pd.read_csv(out/'rainfall_intervals_aligned.csv')
    targets=aggregate_periods(intervals)
    targets.to_csv(out/'diurnal_targets.csv',index=False)
    data=pd.read_csv(out/'dataset_final.csv',parse_dates=['date'])
    fs=json.loads((out/'feature_sets.json').read_text(encoding='utf-8'))
    rows=[]
    for station in ['phuket','krabi','phangnga']:
        for period,lo,hi in PERIODS:
            yy=targets[(targets.station==station)&(targets.period==period)][['date','rain_mm']]
            d=data.merge(yy,on='date',validate='one_to_one').dropna(subset=['rain_mm'])
            tr=d[d.date.dt.year.between(2011,2019)]; te=d[d.date.dt.year.isin([2020,2021])]
            ytr=(tr.rain_mm>=.1).astype(int).to_numpy(); yte=(te.rain_mm>=.1).astype(int).to_numpy()
            if len(np.unique(ytr))<2 or len(np.unique(yte))<2:
                raise ValueError(f'Cannot evaluate both rainfall classes: {station}/{period}')
            for name in MODELS:
                pred,*_=fit_predict(name,tr,te,fs['F4'],fs['F1'],ytr)
                rows.append(dict(station=station,period=period,model=name,feature_set='F4',n_train=len(tr),n_test=len(te),rain_rate_test=float(yte.mean()),roc_auc=roc_auc_score(yte,pred),pr_auc=average_precision_score(yte,pred),brier=brier_score_loss(yte,pred)))
    result=pd.DataFrame(rows,columns=columns); result.to_csv(out/'diurnal_results.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,station in zip(axes,['phuket','krabi','phangnga']):
        for name in MODELS:
            q=result[(result.station==station)&(result.model==name)].set_index('period').loc[[p[0] for p in PERIODS]]
            ax.plot(['07–13','13–19','19–01 (+1d)'],q.roc_auc,marker='o',label=name)
        ax.set_title(station); ax.set_ylim(0,1); ax.grid(alpha=.25); ax.tick_params(axis='x',rotation=15)
    axes[0].set_ylabel('Holdout ROC-AUC'); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir/'fig_diurnal.png',dpi=300); plt.close(fig)
    (out/'diurnal_status.json').write_text(json.dumps({'status':'complete','periods':PERIODS,'accumulation_hours':6,'metadata_provenance':timing.get('provenance')},indent=2),encoding='utf-8')
    print(result.to_string(index=False))


if __name__=='__main__':
    main()
