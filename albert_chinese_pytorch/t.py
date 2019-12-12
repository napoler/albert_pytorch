from albert_pytorch import *

model_name_or_path="prev_trained_model/terry_output/"
model,tokenizer,config_class=Plus().load_model(class_name="AlbertForSequenceClassification",model_path=model_name_or_path)








def train( train_dataloader, model, tokenizer):
    """ Train the model """
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
    #                               collate_fn=collate_fn)

    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
    #     t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    adam_epsilon=1e-8
    learning_rate=5e-5
    warmup_proportion=0.1
    fp16=False
    fp16_opt_level="O1"
    max_grad_norm=1.0
    local_rank=-1
    device='cpu'
    seed=42
    num_train_epochs=10
    n_gpu=0
    t_total=10
    weight_decay=0
    warmup_steps = int(t_total * warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)

    # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)
 
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(int(num_train_epochs)):
        print("111")
        # pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')

        for step, batch in enumerate(train_dataloader):
            print(global_step)
            model.train()
            # batch = tuple(t.to(device) for t in batch)
            # inputs = {'input_ids': batch[0],
            #           'attention_mask': batch[1],
            #           'labels': batch[3]}
            # if args.model_type != 'distilbert':
            # inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'albert',
            #                                                                'roberta'] else None  # XLM, DistilBERT don't use segment_ids
        #    inputs['token_type_ids'] = batch[2]
            input_ids,token_type_ids=Plus().encode(text=batch['text'],tokenizer=tokenizer,max_length=64)
            inputs = {'input_ids': input_ids,
            'attention_mask': input_ids,
            'labels': batch['labels'],
            "token_type_ids":token_type_ids}
            print(inputs)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            print(loss)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            # 验证保存
            # if local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
            #     #Log metrics
            #     if local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
            #         results = evaluate(args, model, tokenizer)

            # if local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
            #     # Save model checkpoint
            #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            #     if not os.path.exists(output_dir):
            #         os.makedirs(output_dir)
            #     model_to_save = model.module if hasattr(model,
            #                                             'module') else model  # Take care of distributed/parallel training
            #     model_to_save.save_pretrained(output_dir)
            #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            #     logger.info("Saving model checkpoint to %s", output_dir)
            #     tokenizer.save_vocabulary(vocab_path=output_dir)
            pbar(step, {'loss': loss.item()})
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step
train_dataloader=[{"text":"这是个牛逼的测试","labels":1},{"text":"这是个","labels":0},{"text":"这是牛逼的测试","labels":1}]
train( train_dataloader, model, tokenizer)