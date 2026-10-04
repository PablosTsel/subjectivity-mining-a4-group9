"""Recompute Block 3 analysis from saved predictions; no training required."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from hatebert_block3 import compare_results, file_hash


def analyze(root=".", output_dir="hatebert_results"):
    root = Path(root)
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    test = pd.read_csv(root / "olid-test.csv")
    metrics = compare_results(out)
    verification, length_rows, error_rows = [], [], []
    results = {}
    for setting in ["in_domain", "cross_domain"]:
        r = pd.read_csv(out / f"results_hatebert_{setting}.csv")
        results[setting] = r
        assert r[["id", "text", "labels"]].equals(test[["id", "text", "labels"]]), "Test rows or order changed"
        assert len(r) == 860 and r.prediction.isin([0, 1]).all()
        assert np.array_equal(r.prediction, r[["logit_0", "logit_1"]].to_numpy().argmax(axis=1))
        assert np.array_equal(r.correct, r.labels == r.prediction)
        assert np.isfinite(r[["logit_0", "logit_1", "probability_offensive"]]).all().all()
        assert r.probability_offensive.between(0, 1).all()
        saved = pd.read_csv(out / f"metrics_hatebert_{setting}.csv").iloc[0]
        p, recall, f, _ = precision_recall_fscore_support(r.labels, r.prediction, average="macro", zero_division=0)
        for key, value in dict(macro_precision=p, macro_recall=recall, macro_f1=f, accuracy=accuracy_score(r.labels,r.prediction)).items():
            assert np.isclose(saved[key], value, atol=1e-12)
        pc, rc, fc, support = precision_recall_fscore_support(r.labels, r.prediction, labels=[0, 1], zero_division=0)
        for label in [0, 1]:
            for key, value in [(f"precision_{label}",pc[label]),(f"recall_{label}",rc[label]),(f"f1_{label}",fc[label]),(f"support_{label}",support[label])]:
                assert np.isclose(saved[key], value, atol=1e-12)
        cm = confusion_matrix(r.labels, r.prediction, labels=[0, 1])
        assert np.array_equal(cm,pd.read_csv(out / f"confusion_hatebert_{setting}.csv",index_col=0).values)
        manifest=json.loads((out / f"run_hatebert_{setting}.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        for name,digest in manifest["data_sha256"].items():
            assert file_hash(root/name) == digest
        verification.append({"setting":setting,"rows_verified":len(r),"metrics_recomputed":True,"data_hashes_match":True,"confusion_matrix_matches":True})
        tn,fp,fn,tp=cm.ravel()
        error_rows.append({"setting":setting,"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"false_positive_rate":fp/(tn+fp),"false_negative_rate":fn/(fn+tp),"predicted_offensive_percent":100*(tp+fp)/len(r)})
        lengths=r.text.str.split().str.len()
        groups=pd.cut(lengths,bins=[-1,15,30,float("inf")],labels=["0-15 words","16-30 words","31+ words"])
        for group in groups.cat.categories:
            s=r[groups==group]
            group_cm=confusion_matrix(s.labels,s.prediction,labels=[0,1])
            length_rows.append({"setting":setting,"length_group":group,"n":len(s),"offensive_support":int((s.labels==1).sum()),"errors":int((~s.correct).sum()),"error_rate":float((~s.correct).mean()),"false_positives":int(group_cm[0,1]),"false_negatives":int(group_cm[1,0])})
    pd.DataFrame(error_rows).to_csv(out/'error_summary_hatebert.csv',index=False)
    pd.DataFrame(length_rows).to_csv(out/'errors_by_length_hatebert.csv',index=False)
    # Paired outcomes refer to the same 860 test examples.
    a,b=results['in_domain'],results['cross_domain']
    paired=a[['id','text','labels']].copy()
    paired['prediction_in_domain']=a.prediction
    paired['prediction_cross_domain']=b.prediction
    paired['outcome']=np.select([a.correct & b.correct,a.correct & ~b.correct,~a.correct & b.correct],["both_correct","only_in_domain_correct","only_cross_domain_correct"],default="both_wrong")
    paired.to_csv(out/'paired_predictions_hatebert.csv',index=False)
    paired.outcome.value_counts().rename_axis('outcome').reset_index(name='n').to_csv(out/'paired_outcomes_hatebert.csv',index=False)
    baseline=pd.concat([pd.read_csv(root/'metrics_logreg_in_domain.csv'),pd.read_csv(root/'metrics_logreg_cross_domain.csv')],ignore_index=True)
    combined=pd.concat([baseline,metrics],ignore_index=True)
    combined.to_csv(out/'metrics_available_models.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    x=np.arange(2); width=.32
    for offset,model,color in [(-width/2,'Logistic Regression','#8797A6'),(width/2,'HateBERT','#28688C')]:
        vals=combined[combined.model==model].set_index('setting').loc[['In-domain','Cross-domain'],'macro_f1'].values
        bars=ax.bar(x+offset,vals,width,label=model,color=color)
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=10)
    ax.set_xticks(x,['In-domain','Cross-domain']); ax.set_ylim(0,1)
    ax.set_xlabel('Training/evaluation setting');ax.set_ylabel('Macro-F1 (0-1)')
    ax.set_title('Available models on the same OLID test set (n = 860)')
    ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(out/'comparison_available_models.png',dpi=180);plt.close(fig)
    (out/'verification.json').write_text(json.dumps({'status':'passed','checks':verification},indent=2),encoding='utf-8')
    print('Verified both prediction files, every metric, confusion matrices and source data hashes.')
    return combined


if __name__ == '__main__':
    print(analyze().to_string(index=False))
