"""Adapt the saved BERT router to fresh training answers without changing splits."""

import hashlib
import json
from pathlib import Path
import random

from .bert_data import LABELS, LABEL_MAP, classifier_text, dataset_hash


def runtime_target(original_label, answer):
    """A refusal despite the constructed clean evidence needs regeneration."""
    from src.detection.response_check import is_refusal
    return 'grounding_failure' if original_label == 'clean' and is_refusal(answer) else original_label


def adapt_runtime_classifier(data, split, generated_train, validation_records, initial_dir,
                             output_dir, seed=20261020, epochs=2):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import BertForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from sklearn.metrics import f1_score, confusion_matrix
    import config
    assert split['dataset_sha256'] == dataset_hash(data)
    train_ids=set(split['indices']['train'])
    assert generated_train and all(r['sample_index'] in train_ids for r in generated_train)
    assert len({r['sample_index'] for r in generated_train}) == len(generated_train)
    assert all(r['sample_index'] in split['indices']['validation'] for r in validation_records)
    assert torch.cuda.is_available(), 'GPU required for the adaptation pass.'
    output_dir=Path(output_dir);initial_dir=Path(initial_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Keep previous checkpoints; use a new empty output directory.')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    tokenizer=AutoTokenizer.from_pretrained(initial_dir,local_files_only=True)
    model=BertForSequenceClassification.from_pretrained(initial_dir,local_files_only=True).to('cuda')
    model.gradient_checkpointing_enable()
    collator=DataCollatorWithPadding(tokenizer)
    max_length=config.BERT_CLASSIFIER_MAX_LENGTH
    def encoded(query,contexts,answer,label):
        value=tokenizer(classifier_text(query,contexts,answer),truncation=True,max_length=max_length)
        value['labels']=LABEL_MAP[label]
        return value
    stored=[encoded(data[i]['query'],data[i]['context_chunks'],data[i]['answer'],data[i]['label']) for i in sorted(train_ids)]
    live=[encoded(data[r['sample_index']]['query'],data[r['sample_index']]['context_chunks'],r['answer'],
                  runtime_target(data[r['sample_index']]['label'],r['answer'])) for r in generated_train]
    # Give the new answer distribution enough weight without discarding the
    # original task. This multiplicity is fixed before validation is examined.
    train=DataLoader(stored+live*4,batch_size=4,shuffle=True,collate_fn=collator,
                     generator=torch.Generator().manual_seed(seed))
    live_val=[encoded(r['input']['query'],r['initial_contexts'],r['initial_answer'],
                     runtime_target(r['input']['label'],r['initial_answer'])) for r in validation_records]
    stored_val=[encoded(data[i]['query'],data[i]['context_chunks'],data[i]['answer'],data[i]['label'])
                for i in split['indices']['validation']]
    def evaluate(active,rows):
        active.eval();truth=[];pred=[]
        with torch.inference_mode():
            for batch in DataLoader(rows,batch_size=8,collate_fn=collator):
                batch={k:v.to('cuda') for k,v in batch.items()}
                logits=active(**batch).logits
                if not torch.isfinite(logits).all():raise ValueError('Non-finite validation logits')
                truth.extend(batch['labels'].cpu().tolist());pred.extend(logits.argmax(-1).cpu().tolist())
        return {'macro_f1':float(f1_score(truth,pred,labels=list(range(5)),average='macro',zero_division=0)),
                'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(5))).tolist(),
                'samples':len(rows)}
    before={'stored':evaluate(model,stored_val),'live':evaluate(model,live_val)}
    output_dir.mkdir(parents=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    history=[];best=-1
    for epoch in range(epochs):
        model.train();optimizer.zero_grad(set_to_none=True);loss_total=0
        for step,batch in enumerate(train):
            batch={k:v.to('cuda') for k,v in batch.items()}
            loss=model(**batch).loss
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            (loss/2).backward();loss_total+=float(loss.detach())
            if (step+1)%2==0 or step+1==len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
        scores={'stored':evaluate(model,stored_val),'live':evaluate(model,live_val)}
        joint=(scores['stored']['macro_f1']+scores['live']['macro_f1'])/2
        history.append({'epoch':epoch+1,'mean_training_loss':loss_total/len(train),'validation':scores,'joint_macro_f1':joint})
        print('BERT_RUNTIME_EPOCH',json.dumps(history[-1]),flush=True)
        if joint>best:
            best=joint;model.save_pretrained(output_dir,safe_serialization=True);tokenizer.save_pretrained(output_dir)
    del model,optimizer
    import gc
    gc.collect();torch.cuda.empty_cache()
    model=BertForSequenceClassification.from_pretrained(output_dir).to('cuda')
    after={'stored':evaluate(model,stored_val),'live':evaluate(model,live_val)}
    # Do not deploy a checkpoint merely because it was trained.
    accepted=(after['live']['macro_f1']>before['live']['macro_f1'] and
              after['stored']['macro_f1']>=before['stored']['macro_f1']-.02)
    old=json.loads((initial_dir/'training_manifest.json').read_text(encoding='utf-8'))
    manifest=dict(old,checkpoint_sha256=hashlib.sha256((output_dir/'model.safetensors').read_bytes()).hexdigest(),
        adaptation={'initial_checkpoint_sha256':old['checkpoint_sha256'],'seed':seed,'epochs':epochs,
            'learning_rate':1e-5,'physical_batch_size':4,'gradient_accumulation':2,
            'generated_training_rows':len(generated_train),'training_original_ids':[r['sample_index'] for r in generated_train],
            'generated_row_multiplicity':4,'training_state_sha256':hashlib.sha256(json.dumps(generated_train,sort_keys=True).encode()).hexdigest(),
            'label_policy':'Original constructed condition; clean context with refusal maps to grounding_failure',
            'before_validation':before,'after_validation':after,'history':history,'accepted':accepted,
            'selection':'Validation only; live macro F1 must improve; stored macro F1 may drop at most 0.02'},
        training_inputs='Original training triples plus fresh generated answers from training IDs only; no validation/test rows in optimisation',
        held_out_test=None)
    (output_dir/'training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output_dir/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    del model
    gc.collect();torch.cuda.empty_cache()
    return manifest
