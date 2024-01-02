import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
import sys
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer
from modules.optimization import AdamW, get_cosine_schedule_with_warmup


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None


    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.huggingface:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16", TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)
    
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented
      
    params_optimizer = list(model.named_parameters())

    clip_params = [n for n, p in params_optimizer if "clip." in n]
    noclip_params = [n for n, p in params_optimizer if "clip." not in n]

    # optimizer_grouped_params = [
    #     {'params': clip_params, 'lr': config.clip_lr},
    #     {'params': noclip_params, 'lr': config.noclip_lr}
    # ]    

    # TODO 冻结CLIP的参数
    flag = False
    for n, p in params_optimizer:
        if "learnable_prompt" in n or "prompt_encoder" in n or "encoder_layer" in n:
            continue
        elif n in noclip_params:
            p.requires_grad = True
        else:
            p.requires_grad = False


    trained_params = [(n,p) for n, p in params_optimizer if p.requires_grad]
    print("The parameters that needed to be trained")
    for n, p in trained_params:
        print(n, p.shape)
    
    optimizer_grouped_params = [
        {'params': [p for n, p in trained_params], 'lr': config.noclip_lr},
    ]
    ppp = get_parameter_number(model)
    print(ppp, ppp["Trainable"]/ppp["Total"])
    # exit()
    # TODO 冻结CLIP的参数
    
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    
    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer,
                      tokenizer=tokenizer)

    trainer.train()


if __name__ == '__main__':
    main()
