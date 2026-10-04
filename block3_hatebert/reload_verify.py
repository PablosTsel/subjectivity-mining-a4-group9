"""Check saved HateBERT checkpoints against all delivered test predictions.

Run after extracting the optional model archive beside this script:
    python reload_verify.py
Uses CUDA and the delivered precision settings, without fitting any model.
"""
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
import torch
from hatebert_block3 import file_hash
from simpletransformers.classification import ClassificationModel


def verify(models_dir="hatebert_models", results_dir="hatebert_results", data_dir="."):
    assert torch.cuda.is_available(), "Enable a CUDA GPU to verify the original inference settings."
    test=pd.read_csv(Path(data_dir)/'olid-test.csv')
    checks=[]
    for setting in ['in_domain','cross_domain']:
        model_dir=Path(models_dir)/setting
        out=Path(results_dir)
        manifest=json.loads((out/f'run_hatebert_{setting}.json').read_text(encoding='utf-8'))
        for name,digest in manifest['model_files_sha256'].items():
            assert file_hash(model_dir/name)==digest, f'Model file changed: {name}'
        model=ClassificationModel('bert',str(model_dir),use_cuda=True,args={
            'use_multiprocessing':False,'use_multiprocessing_for_evaluation':False,
            'fp16':True,'eval_batch_size':32,'max_seq_length':128,
            'reprocess_input_data':True,'no_cache':True,'silent':True,
            'thread_count':4,'wandb_project':None,
        })
        predicted,logits=model.predict(test.text.tolist())
        original=pd.read_csv(out/f'results_hatebert_{setting}.csv')
        equal=np.array_equal(predicted,original.prediction.to_numpy())
        maximum=float(np.max(np.abs(logits-original[['logit_0','logit_1']].to_numpy())))
        assert equal, f'Saved checkpoint predictions differ for {setting}'
        checks.append({'setting':setting,'rows':len(test),'all_predictions_match':bool(equal),'max_absolute_logit_difference':maximum,'model_hashes_match':True})
        print(checks[-1],flush=True)
        del model
        gc.collect();torch.cuda.empty_cache()
    (Path(results_dir)/'reload_verification.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2),encoding='utf-8')
    return checks


if __name__=='__main__':
    verify()
