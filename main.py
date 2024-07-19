# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
import sys
from icecream import ic
sys.path.append('/data/liuyuxuan/SI-T2S/ABSA-QUAD/transformers/src/transformers')

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch.nn.functional as F

# from t5ForPrefixTuning import MyT5ForConditionalGeneration

from transformers import T5ForConditionalGeneration

from transformers import AdamW, T5Tokenizer, AutoTokenizer, RobertaModel, T5Config, RobertaConfig

from peft import get_peft_model, PrefixTuningConfig, TaskType, PeftModel
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import read_line_examples_from_acos
from eval_utils import compute_scores


logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    # parser.add_argument("--task", default='asqp', type=str, required=True,
    #                     help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='rest16', type=str, required=True,
                        help="The name of the dataset, selected from: [laptop14, rest16]")
    parser.add_argument("--bert_model_name_or_path",
                        default='roberta-base',
                        type=str,
                        help="Path to pre-trained model or shortcut name")
    
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_sent_flag", action='store_true',
                        help="Whether to use sent.")
    parser.add_argument("--use_prompt_flag", action='store_true',
                        help="Whether to use prompt.")
    parser.add_argument("--use_augmentation", action='store_true',
                        help="Whether to use LLM augment datset.")
    parser.add_argument("--prefix_tuning", action='store_true',
                        help="Whether to use prefix tuning.")
    parser.add_argument("--dynamic", action='store_true',
                        help="Whether to use dynamic context prefix tuning.")
    parser.add_argument("--token_length", default=128, type=int)
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")
    # other parameters
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--E_I", default='ALL')
    parser.add_argument("--cate_type", default=-1, type=int)
    parser.add_argument("--num_clusters", default=-1, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest16/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"outputs/{args.dataset}"
    # output_dir = f"outputs_NO_LM/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    output_dir_adapter = f'./outputs_adapter'
    # output_dir_adapter = f'./outputs_adapter_LLM'
    if not os.path.exists(output_dir_adapter):
        os.mkdir(output_dir_adapter)
    if args.cate_type != -1:
        adapter_dir = f"{output_dir_adapter}/{args.dataset}/{args.token_length}_{args.cate_type}"
    else:
        adapter_dir = f"{output_dir_adapter}/{args.dataset}"

    if not os.path.exists(adapter_dir):
        os.mkdir(adapter_dir)
    args.adapter_dir = adapter_dir
    
    
    return args



def get_dataset(tokenizer, data_dir, type_path, args):
    # print(args.use_prompt_flag,args.use_sent_flag)
    return ABSADataset(tokenizer=tokenizer,
                       data_dir=args.dataset,
                       data_type=type_path,
                       max_len=args.max_seq_length,
                       use_sent_flag=args.use_sent_flag,
                       use_prompt_flag=args.use_prompt_flag,
                       use_augmentation=args.use_augmentation,
                       cate_type=args.cate_type,
                       num_clusters=args.num_clusters)

def context_attention(x, y):
    device = torch.device(f'cuda:{args.n_gpu}')
    # 将 vir_embedding [128,] 的维度扩展，以便与 cls [b,786] 进行 matmul
    y_trans = nn.Linear(t5_config.d_model, args.token_length * 2, bias=True).to(device)
    y = y_trans(y).to(device) # [b,786] -> [b,128]

    x_expanded = x.unsqueeze(0).expand_as(y).to(device).to(dtype=torch.float32)
    scores = torch.matmul(x_expanded, y.T).to(device)
    # 使用 softmax 将分数转换为注意力权重
    weights = F.softmax(scores, dim=-1).to(device)
    # 将注意力权重应用于向量 y
    attended_vector = torch.matmul(weights, y).to(device)
    attended_vector = torch.sum(attended_vector, dim=0).to(device) / attended_vector.size(0)

    return attended_vector, weights

class MultiPrefixContextAttention(nn.Module):
    def __init__(self, d_model=768, config=None):
        super(MultiPrefixContextAttention, self).__init__()
        self.d_model = d_model
        # embedding = 18432
        # clusters = 6
        self.device = torch.device(f'cuda:{args.n_gpu}')
        self.W_q = nn.Linear(self.d_model * 2 * 12 , 768, bias=True).to(self.device)
        self.W_k = nn.Linear(768, 768, bias=True).to(self.device)
        self.W_v = nn.Linear(768, 768, bias=True).to(self.device)

        self.W_o = nn.Linear(768, 768, bias=True).to(self.device)

    def forward(self, query, key, value):
        # query [6, 64, 18432] -> [6, 64, 768]
        # ic(self.device)
        Q = self.W_q(query).to(self.device)
        # key [8, 768] -> [8, 768]
        K = self.W_k(key).to(self.device)
        # value [8, 768] -> [8, 768]
        V = self.W_v(value).to(self.device)
        # dot_product [6, 64, 8]
        dot_product = torch.matmul(Q, K.transpose(0, 1)).to(self.device)
        # scaled sqrt(d_k)
        scaled_dot_product = dot_product / torch.sqrt(torch.tensor(768.0)).to(self.device)
        # softmax
        attention_weights = F.softmax(scaled_dot_product, dim=-1).to(self.device)
        # attention_value [6, 64, 768]
        attention_value = torch.matmul(attention_weights, value).to(self.device)

        attention_outputs = self.W_o(attention_value).to(self.device)
        # attention_outputs = torch.mean(attention_outputs, dim=0).to(self.device)
        # attention_outputs = torch.sum(attention_outputs, dim=0).to(self.device)
        return attention_outputs

class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, args, tfm_model, tokenizer, bert_tokenizer=None, bert_model=None, prefix_tuning_flag=None, dynamic_flag=None):
        super().__init__()
        self.args = args
        self.prefix_tuning_flag = prefix_tuning_flag
        self.dynamic_flag = dynamic_flag
        self.model_adapter_list = []
        self.cate_prefix_embeddings = []
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.multi_prefix_context_attention = MultiPrefixContextAttention()
        self.prefix_query = None
        
        # 加载prefix-encoder
        if self.prefix_tuning_flag:
            peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                                 num_virtual_tokens=args.token_length,
                                 token_dim=t5_config.d_model,
                                 encoder_hidden_size=t5_config.d_model,
                                 num_layers=t5_config.num_layers,
                                 num_attention_heads=t5_config.num_heads,
                                 inference_mode=False,
                                 prefix_projection=True)
            self.model = get_peft_model(tfm_model, peft_config)
            ic("********* use prefix-tuning *********")
            self.model.print_trainable_parameters()
            # 查看动态prefix
            # self.model.prompt_encoder['default'].transform
            # ic(self.model.prompt_encoder['default'].embedding.weight.size())
            
            if self.dynamic_flag:
                ic("********* use dynamic with context attention prefix-tuning *********")
                # need change
                # --------------------------------
                adapter_paths = [f"/data/liuyuxuan/SI-T2S/ABSA-QUAD-V2/outputs_adapter_LLM/{args.dataset}/{args.token_length}_{x}" for x in range(0, args.
                num_clusters)]
                ic(adapter_paths)
                
                # check dir
                for cur_path in adapter_paths:
                    if os.path.exists(cur_path) == False:
                        print(f"wrong adapter path : {cur_path}")
                        sys.exit(0)
                for idx, item in enumerate(adapter_paths):
                    # tfm_model !
                    self.model_adapter_list.append(PeftModel.from_pretrained(tfm_model, item, adapter_name=f"{args.dataset}_prefix_{idx}"))
                assert args.num_clusters == len(self.model_adapter_list)

                for idx, model_a in enumerate(self.model_adapter_list):
                    self.cate_prefix_embeddings.append(model_a.prompt_encoder[f'{args.dataset}_prefix_{idx}'].embedding.weight)
                if self.cate_prefix_embeddings == None:
                    print("error !")
                    sys.exit(0)

                self.prefix_query = torch.stack(self.cate_prefix_embeddings, dim=0)
        else:
            self.model = tfm_model


        self.tokenizer = tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.bert2t5_transform = nn.Sequential(
                                 nn.Linear(in_features=t5_config.d_model, out_features=bert_config.hidden_size, bias=True),
                                 nn.Tanh())

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        # get cls    (bs, h)
        if self.prefix_tuning_flag and self.dynamic_flag:
            context_cls = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
        #     # [b, h]
            context_cls = self.bert2t5_transform(context_cls)
        #     # attention with context
            
            context_cls.to("cuda:0")
            self.prefix_query = self.prefix_query.to(context_cls.device)
            # ic(context_cls.device)
            # ic(self.prefix_query.device)

            output = self.multi_prefix_context_attention(query=torch.sum(self.prefix_query, dim=0), 
                                                         key=context_cls, 
                                                         value=context_cls)
            output_parameter = torch.nn.Parameter(output)
            # ic(self.model.prompt_encoder['default'].embedding.weight.size())
            # ic(output.size())
            self.model.prompt_encoder['default'].embedding.weight.data +=  output_parameter.data
            
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )
        
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.training_step_outputs.append(loss)
        tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}
    # on_train_epoch_end

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.training_step_outputs).mean()
        # avg_train_loss = torch.stack(pl_module.training_step_outputs).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.training_step_outputs.clear()
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}
    # on_validation_epoch_end
    def on_validation_epoch_end(self):
        # avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        self.validation_step_outputs.clear()
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0001,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    #     if self.trainer.accelerator=='tpu':
    #         xm.optimizer_step(optimizer)
    #     else:
    #         optimizer.step()
    #     optimizer.zero_grad()
    #     self.lr_scheduler.step()
    
    # def optimizer_step(self,
    #                  epoch=None,
    #                  batch_idx=None,
    #                  optimizer=None,
    #                  optimizer_idx=None,
    #                  optimizer_closure=None,
    #                  on_tpu=None,
    #                  using_native_amp=None,
    #                  using_lbfgs=None):

    #     optimizer.step() # remove 'closure=optimizer_closure' here
    #     optimizer.zero_grad()
    #     self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=args.dataset ,type_path="train", args=self.args)
        print("train dataset size:", len(train_dataset))
        dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=1)
        t_total = (
            (len(dataloader.dataset) // (self.args.train_batch_size * max(1, len(self.args.n_gpu))))
            // self.args.gradient_accumulation_steps
            * float(self.args.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=args.dataset, type_path="dev", args=self.args)
        return DataLoader(val_dataset, batch_size=self.args.eval_batch_size, num_workers=1)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, sents):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu}')
    model.model.to(device)

    model.model.eval()

    outputs, targets = [], []

    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=args.max_seq_length)  
        # num_beams=8, early_stopping=True)

        dec = [
            tokenizer.decode(ids, skip_special_tokens=True) 
            for ids in outs
            ]
        target = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
            ]

        outputs.extend(dec)
        targets.extend(target)

    '''
    print("\nPrint some results to check the sanity of generation method:", '\n', '-'*30)
    for i in [1, 5, 25, 42, 50]:
        try:
            print(f'>>Target    : {targets[i]}')
            print(f'>>Generation: {outputs[i]}')
        except UnicodeEncodeError:
            print('Unable to print due to the coding error')
    print()
    '''

    scores, all_labels, all_preds = compute_scores(outputs, targets, sents, args.use_sent_flag, args.use_prompt_flag)
    results = {'scores': scores, 'labels': all_labels, 'preds': all_preds}
    # pickle.dump(results, open(f"{args.output_dir}/results-{args.dataset}.pickle", 'wb'))

    return scores


if __name__ == '__main__':
    # initialization
    print(f"注意当前的pid: {os.getpid()}")
    args = init_args()
    print("\n", "="*30, f"NEW EXP: ACOS on {args.dataset}", "="*30, "\n")

    # bert_path = r"D:\ABSA-QUAD\model_dir\roberta_base"
    bert_path = "/data/liuyuxuan/SI-T2S/ABSA-QUAD-V2/model_dir/roberta_base"
    # t5_path = r"D:\ABSA-QUAD\model_dir\t5_base"
    t5_path = "/data/liuyuxuan/SI-T2S/ABSA-QUAD-V2/model_dir/t5_base"
    seed_everything(args.seed)

    # sanity check
    # show one sample to check the code and the expected output
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    # tokenizer.add_special_tokens({"additional_special_tokens":['[aspect]', '[category]', '[opinion]', '[sentiment]']})

    print(f"Here is an example (from the dev set):")
    dataset = ABSADataset(tokenizer=tokenizer,
                        data_dir=args.dataset, 
                        data_type='dev', 
                        max_len=args.max_seq_length, 
                        use_sent_flag=args.use_sent_flag,
                        use_prompt_flag=args.use_prompt_flag,
                        use_augmentation=args.use_augmentation)
    data_sample = dataset[7]  # a random data sample
    print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))



    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = RobertaModel.from_pretrained(bert_path)
    bert_config = RobertaConfig.from_pretrained(bert_path)

    t5_config = T5Config.from_pretrained(t5_path)

    # training process
    if args.do_train:
        print("\n****** Conduct Training ******")

        # initialize the T5 model
        if args.prefix_tuning:
            ic("############ reload fine-tuning model ############")
            tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)
        else:
            tfm_model = T5ForConditionalGeneration.from_pretrained(t5_path)
    
        model = T5FineTuner(args, tfm_model, tokenizer, bert_tokenizer, bert_model, prefix_tuning_flag=args.prefix_tuning,
                            dynamic_flag=args.dynamic)
        # save_top_k = -1
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir, 
            filename="ckt", 
            monitor='val_loss', 
            mode='min', 
            save_top_k=3
        )

        # prepare for trainer
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            devices=[int(args.n_gpu)],
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            callbacks=[LoggingCallback()],
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        if args.prefix_tuning:
            model.model.save_pretrained(args.adapter_dir)
        else:
            model.model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

        print("Finish training and saving the model!")


    if args.do_direct_eval:
        print("\n****** Conduct Evaluating with the last state ******")

        # tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        # model = T5FineTuner(args, tfm_model, tokenizer)

        print("Reload the model to do direct eval")
        tokenizer = T5Tokenizer.from_pretrained(t5_path)
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)
        px_model = T5FineTuner(args, tfm_model, tokenizer, bert_tokenizer, bert_model, prefix_tuning_flag=False)
        if args.prefix_tuning:
            print(f"Load trained model from {args.adapter_dir}")
            ic(args.adapter_dir)
            ic(args.output_dir)
            px_model.model = PeftModel.from_pretrained(tfm_model, args.adapter_dir, adapter_name=f"{args.dataset}_prefix")

        sents, _ = read_line_examples_from_acos(f'data/acos/{args.dataset}/test.tsv')

        print()
        E_I_list = ["ALL", "EAEO", "EAIO", "IAEO", "IAIO"]
        # E_I_list = ["ALL"]
        exp_results_list = []
        for ei in E_I_list:
            print("it is {}".format(ei))
            test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                                data_type='test', max_len=args.max_seq_length,
                                E_I=ei,
                                use_sent_flag=args.use_sent_flag,
                                use_prompt_flag=args.use_prompt_flag)
            test_loader = DataLoader(test_dataset, batch_size=32, num_workers=1)
            # print(test_loader.device)

            # compute the performance scores
            scores = evaluate(test_loader, px_model, sents)

            # write to file
            log_file_path = f"results_log/{args.dataset}.txt"
            local_time = time.asctime(time.localtime(time.time()))

            exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
            exp_results = f"F1 = {scores['f1']:.4f}"
            exp_results_list.append(exp_results)
        exp_results_str = ", ".join(exp_results_list)
        log_str = f'============================================================\n'
        log_str += f"{local_time}\n{exp_settings}\n{exp_results_str}\n\n"

        if not os.path.exists('./results_log'):
            os.mkdir('./results_log')

        with open(log_file_path, "a+") as f:
            f.write(log_str)
        ic("==================== done! ====================")



    if args.do_inference:
        print("\n****** Conduct inference on trained checkpoint ******")

        # initialize the T5 model from previous checkpoint
        print('Note that a pretrained model is required and `do_true` should be False')
        tokenizer = T5Tokenizer.from_pretrained(t5_path)
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)
        px_model = T5FineTuner(args, tfm_model, tokenizer, bert_tokenizer, bert_model, prefix_tuning_flag=False)
        if args.prefix_tuning:
            print(f"Load trained model from {args.adapter_dir}")
            ic(args.adapter_dir)
            ic(args.output_dir)
            px_model.model = PeftModel.from_pretrained(tfm_model, args.adapter_dir, adapter_name=f"{args.dataset}_prefix")
       

        # model = T5FineTuner(args, tfm_model, tokenizer)
        
        # if args.prefix_tuning:
        #     prefix_config = PeftConfig.from_pretrained(args.output_dir)
        #     model = PeftModel.from_pretrained(tfm_model, args.output_dir)

        sents, _ = read_line_examples_from_acos(f'data/acos/{args.dataset}/test.tsv')

        print()
        test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                                data_type='test', max_len=args.max_seq_length,
                                E_I=args.E_I,
                                use_sent_flag=args.use_sent_flag,
                                use_prompt_flag=args.use_prompt_flag,
                                use_augmentation=args.use_augmentation)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=1)
        # print(test_loader.device)

        # compute the performance scores
        scores = evaluate(test_loader, px_model, sents)

        # write to file
        log_file_path = f"results_log/{args.dataset}.txt"
        local_time = time.asctime(time.localtime(time.time()))

        exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
        exp_results = f"F1 = {scores['f1']:.4f}"

        log_str = f'============================================================\n'
        log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

        if not os.path.exists('./results_log'):
            os.mkdir('./results_log')

        with open(log_file_path, "a+") as f:
            f.write(log_str)
