# simCSE
## Training
- 根據預訓練模型的不同(Bert或RoBerta)決定使用哪一種simcse/models
    -  train.py
        ```python
        if model_args.model_name_or_path:
                if 'roberta' in model_args.model_name_or_path:
                    model = RobertaForCL.from_pretrained(
                        model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        config=config,
                        cache_dir=model_args.cache_dir,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                        model_args=model_args                  
                    )
                elif 'bert' in model_args.model_name_or_path:
                    model = BertForCL.from_pretrained(
                        model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        config=config,
                        cache_dir=model_args.cache_dir,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                        model_args=model_args
                    )
                    if model_args.do_mlm:
                        pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                        model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
                else:
                    raise NotImplementedError
        ```
- loss_function
    - 採用 `nn.CrossEntropyLoss()` 
    - 輸入是 cos_sim 與 labels
        ```python
        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)    
        ```
        - `cl_forward` 中查看詳細的計算方式

## Refference
- [中文淺顯易懂簡介](https://fcuai.tw/2021/05/13/simcsecontrastive-learning-nlp-sentence-embedding-sota/)

