import os
import collections
import logging
import warnings
import random
import time
import pathlib
import json

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn
import torch_geometric

from gqa.gqa_data import GQADataset, GQATorchDataset, GQATorchDataset_collate_fn
from param import args
from gqa.gqa_model import GQAModel
from coarse.pretrain.qa_answer_table import load_lxmert_qa
import utils
from fine.graphvqa.config import graphVQAConfig

from torch.nn import DataParallel as DataParalle_raw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel as DataParallel_raw
import numpy as np


class DataParallel(DataParallel_raw):
    """
    we do the scatter outside of the DataPrallel.
    input: Scattered Inputs without kwargs.
    """

    def __init__(self, module):
        # Disable all the other parameters
        super(DataParallel, self).__init__(module)

    def custom_scatter(self, kwargs):
        if len(self.device_ids) == 2:
            sizes = [64, 64]
        elif len(self.device_ids) == 3:
            sizes = [64, 64, 64]
        elif len(self.device_ids) == 4:
            sizes = [64, 64, 64, 64]
            # sizes = [2, 2, 2, 2]
        new_inputs = [{} for _ in self.device_ids]
        for key in kwargs:
            idx0 = sizes[0]
            idx1 = sizes[0]+sizes[1]
            idx2 = sizes[0]+sizes[1]+sizes[2]
            if key == 'feats' or key == 'boxes':
                new_inputs[0][key] = kwargs[key][:idx0].to(self.device_ids[0])
                new_inputs[1][key] = kwargs[key][idx0:idx1].to(self.device_ids[1])
                if len(self.device_ids) >= 3:
                    new_inputs[2][key] = kwargs[key][idx1:idx2].to(self.device_ids[2])
                if len(self.device_ids) >= 4:
                    new_inputs[3][key] = kwargs[key][idx2:].to(self.device_ids[3])   
            elif key == 'ques':
                new_inputs[0][key] = kwargs[key][:idx0]
                new_inputs[1][key] = kwargs[key][idx0:idx1]
                if len(self.device_ids) >= 3:
                    new_inputs[2][key] = kwargs[key][idx1:idx2]
                if len(self.device_ids) >= 4:
                    new_inputs[3][key] = kwargs[key][idx2:]
            elif key == 'sg':
                new_inputs[0][key] = \
                    torch_geometric.data.Batch.from_data_list(kwargs[key][:idx0]).to(self.device_ids[0])
                new_inputs[1][key] = \
                    torch_geometric.data.Batch.from_data_list(kwargs[key][idx0:idx1]).to(self.device_ids[1])
                if len(self.device_ids) >= 3:
                    new_inputs[2][key] = \
                        torch_geometric.data.Batch.from_data_list(kwargs[key][idx1:idx2]).to(self.device_ids[2])
                if len(self.device_ids) >= 4:
                    new_inputs[3][key] = \
                        torch_geometric.data.Batch.from_data_list(kwargs[key][idx2:]).to(self.device_ids[3])
            elif key == 'ques_fine':
                new_inputs[0][key] = kwargs[key][:, :idx0].to(self.device_ids[0])
                new_inputs[1][key] = kwargs[key][:, idx0:idx1].to(self.device_ids[1])
                if len(self.device_ids) >= 3:
                    new_inputs[2][key] = kwargs[key][:, idx1:idx2].to(self.device_ids[2])
                if len(self.device_ids) >= 4:
                    new_inputs[3][key] = kwargs[key][:, idx2:].to(self.device_ids[3])
            elif key == 'prog_in':
                e = graphVQAConfig.max_execution_step
                new_inputs[0][key] = kwargs[key][:, :idx0*e].to(self.device_ids[0])
                new_inputs[1][key] = kwargs[key][:, idx0*e:idx1*e].to(self.device_ids[1])
                if len(self.device_ids) >= 3:
                    new_inputs[2][key] = kwargs[key][:, idx1*e:idx2*e].to(self.device_ids[2])
                if len(self.device_ids) >= 4:
                    new_inputs[3][key] = kwargs[key][:, idx2*e:].to(self.device_ids[3])
            elif key == 'val':
                new_inputs[0][key] = kwargs[key]
                new_inputs[1][key] = kwargs[key]
                if len(self.device_ids) >= 3:
                    new_inputs[2][key] = kwargs[key]
                if len(self.device_ids) >= 4:
                    new_inputs[3][key] = kwargs[key]
            
        return new_inputs

    def forward(self, *inputs, **kwargs):
        #assert len(inputs) == 0, "Only support arguments like [variable_name = xxx]"
        #new_inputs = [{} for _ in self.device_ids]
        nones = [[] for _ in self.device_ids]
        #print (self.device_ids)
        new_inputs = self.custom_scatter(kwargs)
        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, nones, new_inputs)
        return self.gather(outputs, self.output_device)


DataTuple = collections.namedtuple("DataTuple", 'dataset tdataset loader sampler')
cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Default CUDA device

def get_tuple(split: str, bs:int) -> DataTuple:
    dset = GQADataset(split)
    tset = GQATorchDataset(dset)
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(tset)
    else:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(tset)
        else:
            sampler = torch.utils.data.SequentialSampler(tset)
    if split == 'train' :
        batch_sampler = torch.utils.data.BatchSampler(sampler, bs, drop_last=True)
        data_loader = DataLoader(
            tset, 
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=GQATorchDataset_collate_fn,
            pin_memory=True
        )
    else:
        data_loader = DataLoader(
            tset, 
            batch_size=bs,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=GQATorchDataset_collate_fn,
            pin_memory=True
        )

    return DataTuple(dataset=dset, tdataset=tset, loader=data_loader, sampler=sampler)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

        ##################################
        # Save to logging
        ##################################
        if utils.is_main_process():
            logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

"""
Input shape: [Len, Batch]
A fast GPU-based string exact match accuracy calculator

TODO: if the prediction does not stop at target's padding area.
(should rarely happen if at all)

group_accuracy_WAY_NUM: only calculated as correct if the whole group is correct.
Used in program accuracy: only correct if all instructions are correct.
-1 means ignore
"""
def program_string_exact_match_acc(predictions, target, padding_idx=1, group_accuracy_WAY_NUM=-1):

    ##################################
    # Do token-level match first
    # Generate a matching matrix: if equals or pad, then put 1, else 0
    # Shape: [Len, Batch]
    ##################################
    target_len = target.size(0)
    # truncated
    predictions = predictions[:target_len]
    char_match_matrix = (predictions == target).long()
    cond_match_matrix = torch.where(target == padding_idx, torch.ones_like(target), char_match_matrix)
    del char_match_matrix

    ##################################
    # Reduction of token-level match
    # 1 means exact match, 0 means at least one token not matching
    # Dim: note that the first dim is len, batch is the second dim
    ##################################
    # ret: (values, indices)
    match_reduced, _ = torch.min(input=cond_match_matrix, dim=0, keepdim=False)
    this_batch_size = target.size(1)
    # mul 100, converting to percentage
    accuracy = torch.sum(match_reduced).item() / this_batch_size * 100.0

    ##################################
    # Calculate Batch Accuracy
    ##################################
    group_batch_size = this_batch_size // group_accuracy_WAY_NUM
    match_reduced_group_reshape = match_reduced.view(group_batch_size, group_accuracy_WAY_NUM)
    # print("match_reduced_group_reshape", match_reduced_group_reshape)
    # ret: (values, indices)
    group_match_reduced, _ = torch.min(input=match_reduced_group_reshape, dim=1, keepdim=False)
    # print("group_match_reduced", group_match_reduced)
    # mul 100, converting to percentage
    group_accuracy = torch.sum(group_match_reduced).item() / group_batch_size * 100.0

    ##################################
    # Calculate Empty
    # start of sentence, end of sentence, padding
    # Shape: [Len=2, Batch]
    ##################################
    # empty and counted as correct
    empty_instr_flag = (target[2] == padding_idx) & match_reduced.bool()
    empty_instr_flag = empty_instr_flag.long()
    # print("empty_instr_flag", empty_instr_flag)
    empty_count = torch.sum(empty_instr_flag).item()
    # print("empty_count", empty_count)
    non_empty_accuracy = (torch.sum(match_reduced).item() - empty_count) / (this_batch_size - empty_count) * 100.0

    ##################################
    # Return
    ##################################
    return accuracy, group_accuracy , non_empty_accuracy


def load_graphvqa(ckpt_path, model):
    loaded_state_dict = torch.load(ckpt_path)['model']
    model_state_dict = model.state_dict()
    for key in list(loaded_state_dict.keys()):
        loaded_state_dict[key.replace("module.", '')] = loaded_state_dict.pop(key)
    mkeys = set([k.replace('fine_encoder.', '') for k in model.fine_encoder.state_dict().keys()])
    lkeys = set(loaded_state_dict.keys())
    new_state_dict = {}
    new_state_dict['fine_logit_fc.4.weight'] = loaded_state_dict['logit_fc.4.weight']
    new_state_dict['fine_logit_fc.4.bias'] = loaded_state_dict['logit_fc.4.bias']
    model.fine_encoder.load_state_dict(loaded_state_dict, strict=False)
    model.load_state_dict(new_state_dict, strict=False)

class GQA:
    def __init__(self):
        self.num_answers = 1842
        if args.train :
            self.train_tuple = get_tuple(
                args.train, bs=args.batch_size
            )
        else:
            self.train_tuple = None

        if args.valid:
            self.valid_tuple = get_tuple(
                args.valid, bs=args.batch_size
            )
        else:
            self.valid_tuple = None

        self.model = GQAModel(self.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.coarse_encoder.load(args.load_lxmert)

        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        if args.load_graphvqa is not None:
            load_graphvqa(args.load_graphvqa, self.model)

        # GPU options
        self.model = self.model.to(device=cuda)

        self.model_without_ddp = self.model
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=os.environ['CUDA_VISIBLE_DEVICES'], find_unused_parameters=True
            )
            self.model_without_ddp = self.model.module
        elif args.multiGPU:
            self.model = DataParallel(self.model)
            self.model_without_ddp = self.model.module
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        # Losses and optimizer
        #self.bce_loss = nn.BCEWithLogitsLoss()
        #self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        '''
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from coarse.lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)
        '''

        if args.train:
            # self.criterion = {
            #     "program": torch.nn.CrossEntropyLoss(ignore_index=graphVQAConfig.pad_token).to(device=cuda),
            #     "label": torch.nn.CrossEntropyLoss(ignore_index=-1).to(device=cuda),
            #     "execution_bitmap": torch.nn.BCELoss().to(device=cuda),
            # }
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device=cuda)
            self.optim = torch.optim.Adam(
                params=self.model.parameters(),
                lr=args.lr
            )
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, args.lr_drop)

        self.output = args.output

    def train(self, train_tuple, eval_tuple):
        dset, tset, loader, sampler = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        print ("Loader : ",len(loader))
        best_valid = 0
        cur_valid = 0
        for epoch in range(args.epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            self.lr_scheduler.step()
            batch_time = AverageMeter('Time', ':4.2f')
            data_time = AverageMeter('Data', ':4.2f')
            losses = AverageMeter('Loss', ':3.5f')
            answer_acc = AverageMeter('Acc', ':4.2f')
            program_acc = AverageMeter('Acc@Program', ':6.2f')
            program_group_acc = AverageMeter('Acc@ProgramGroup', ':4.2f')
            program_non_empty_acc = AverageMeter('Acc@ProgramNonEmpty', ':4.2f')
            progress = ProgressMeter(
                len(loader),
                [
                    batch_time, data_time, losses,
                    #program_acc, program_group_acc, program_non_empty_acc,
                    answer_acc
                ],
                prefix="Epoch: [{}]".format(epoch))
            end = time.time()
            for i, (ques_id, feats, boxes, ques, ques_fine, sg, prog, \
            target, types) in iter_wrapper(enumerate(loader)):
                data_time.update(time.time() - end)
                self.model.train()
                self.optim.zero_grad()
                #target = torch.LongTensor(target)
                target = target.to(device=cuda)
                if not args.multiGPU:
                    feats, boxes, target = feats.to(device=cuda), boxes.to(device=cuda), target.to(device=cuda)
                    ques_fine = ques_fine.to(device=cuda)
                    sg, prog = torch_geometric.data.Batch.from_data_list(sg).to(device=cuda), prog.to(device=cuda)
                batch_size = ques_fine.size(1)
                prog_in = prog[:-1]
                prog_target = prog[1:]
                logit, prog_out = self.model(feats=feats, boxes=boxes, ques=ques, ques_fine=ques_fine, sg=sg, prog_in=prog_in)
                #print (logit.device)
                target = target.to(device=cuda)
                #print (target.device)
                #assert logit.dim() == target.dim() == 2

                with torch.no_grad():
                    answer_acc1 = accuracy(logit, target, topk=(1,))
                    answer_acc.update(answer_acc1[0].item(), batch_size)
                    '''
                    prog_out_pred = prog_out.detach().topk(
                        k=1, dim=-1, largest=True, sorted=True
                    )[1].squeeze(-1)
                    program_acc1, program_group_acc1, program_non_empty_acc1 = program_string_exact_match_acc(
                        prog_out_pred, prog_target,
                        padding_idx=graphVQAConfig.pad_token,
                        group_accuracy_WAY_NUM=graphVQAConfig.max_execution_step)
                    program_acc.update(program_acc1, batch_size)
                    program_group_acc.update(program_group_acc1, batch_size // graphVQAConfig.max_execution_step)
                    program_non_empty_acc.update(program_non_empty_acc1, batch_size)
                    '''
                ans_loss = self.criterion(logit, target)
                loss = ans_loss + 0 * prog_out.mean()
                # loss = ans_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                losses.update(loss.item(), batch_size)
                batch_time.update(time.time() - end)
                end = time.time()

                #if i == 0 or args.print_freq % (i + 1) == 0 or i == len(loader) - 1:
                #    progress.display(i)

            progress.display(batch=len(loader))

            if eval_tuple is not None:  # Do Validation
                cur_valid = self.predict(eval_tuple, "valid")
            
            if args.output:
                #if epoch == 0 or (epoch + 1) % 100 == 0 or cur_valid > best_valid:
                best_valid = cur_valid
                self.save(epoch, best_valid)

    def predict(self, eval_tuple: DataTuple, mode):
        dset, tset, loader, sampler = eval_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        batch_time = AverageMeter('Time', ':6.3f')
        answer_acc = AverageMeter('Acc', ':6.2f')
        program_acc = AverageMeter('Acc@Program', ':6.2f')
        program_group_acc = AverageMeter('Acc@ProgramGroup', ':4.2f')
        program_non_empty_acc = AverageMeter('Acc@ProgramNonEmpty', ':4.2f')
        progress = ProgressMeter(
            len(loader),
            [
                batch_time, 
                #program_acc, program_group_acc, program_non_empty_acc,
                answer_acc
            ],
            prefix='Test: '
        )
        self.model.eval()

        if args.save_result:
            quesid2ans = {}

        with torch.no_grad():
            end = time.time()
            for i, (ques_id, feats, boxes, ques, ques_fine, sg, prog, \
            target, types) in iter_wrapper(enumerate(loader)):
                #target = torch.LongTensor(target)
                #feats, boxes, target = feats.to(device=cuda), boxes.to(device=cuda), target.to(device=cuda)
                #ques_fine, sg, prog = ques_fine.to(device=cuda), sg.to(device=cuda), prog.to(device=cuda)
                target = target.to(device=cuda)
                if not args.multiGPU:
                    feats, boxes, target = feats.to(device=cuda), boxes.to(device=cuda), target.to(device=cuda)
                    ques_fine = ques_fine.to(device=cuda)
                    sg, prog = torch_geometric.data.Batch.from_data_list(sg).to(device=cuda), prog.to(device=cuda)
                batch_size = ques_fine.size(1)
                prog_target = prog
                logit = self.model(feats=feats, boxes=boxes, ques=ques, ques_fine=ques_fine, sg=sg, prog_in=prog, val=True)
                #assert logit.dim() == target.dim() == 2
                #target = target.to(device=cuda)
                answer_acc1 = accuracy(logit, target, topk=(1,))
                answer_acc.update(answer_acc1[0].item(), batch_size)
                '''
                program_acc1, program_group_acc1, program_non_empty_acc1 = program_string_exact_match_acc(
                    prog_out_pred, prog_target,
                    padding_idx=graphVQAConfig.pad_token,
                    group_accuracy_WAY_NUM=graphVQAConfig.max_execution_step
                )
                program_acc.update(program_acc1, batch_size)
                program_group_acc.update(program_group_acc1, batch_size // graphVQAConfig.max_execution_step)
                program_non_empty_acc.update(program_non_empty_acc1, batch_size)
                '''
                if args.save_result:
                    pred_score, pred_label = logit.max(1)
                    pred_score, pred_label = pred_score.cpu(), pred_label.cpu()
                    for batch_idx in range(batch_size):
                        question = ques[batch_idx]
                        qid = ques_id[batch_idx]
                        '''
                        ground_truth_program_list = []
                        predicted_program_list = []
                        for instr_idx in range(graphVQAConfig.max_execution_step):
                            true_batch_idx = instr_idx + graphVQAConfig.max_execution_step * batch_idx
                            gt = prog[:, true_batch_idx].cpu()
                            pred = prog_out_pred[:, true_batch_idx]
                            pred_sent, _ = GQATorchDataset.indices_to_string(pred, True)
                            gt_sent, _ = GQATorchDataset.indices_to_string(gt, True)

                            if len(pred_sent) == 0 and len(gt_sent) == 0:
                                continue

                            ground_truth_program_list.append(gt_sent)
                            predicted_program_list.append(pred_sent)
                        '''
                        quesid2ans[qid] = {
                            "questionId": str(qid),
                            "question": question,
                            #"ground_truth_program_list": ground_truth_program_list,
                            #"predicted_program_list": predicted_program_list,
                            "answer": dset.label2ans[target[batch_idx].cpu().item()],
                            "prediction": dset.label2ans[pred_label[batch_idx].cpu().item()],
                            "prediction_score": '{:.2f}'.format(pred_score[batch_idx].cpu().item()),
                            "types": types[batch_idx]
                        }
                
                batch_time.update(time.time() - end)
                end = time.time()

        progress.display(batch=len(loader))
        if args.save_result:
            result_dump_path = os.path.join(args.output, "dump2_results_"+mode+".json")
            with open(result_dump_path, 'w') as f:
                json.dump(quesid2ans, f, indent=4, sort_keys=True)
                print("Result Dumped!", str(result_dump_path))
        return answer_acc.avg

    def save(self, epoch, acc):
        output_dir = pathlib.Path(args.output)
        checkpoint_path = output_dir / f'checkpoint2_{epoch:03}.pth'
        print("Save model to %s" % checkpoint_path)
        utils.save_on_master(self.model_without_ddp.state_dict(), checkpoint_path)

    def load(self, path):
        output_dir = pathlib.Path(args.output)
        checkpoint_path = output_dir / ("%s.pth" % path)
        print("Load model from %s" % checkpoint_path)
        state_dict = torch.load(checkpoint_path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model_without_ddp.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    from fine.vocab.vocab_utils import QAVocab, SGVocab, TextProcessor, Vocab

    qaVocab = QAVocab(build=False)
    sgVocab = SGVocab(build=False)
    # Build Class
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)


    if args.seed is not None:
        # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    if args.output and utils.is_main_process():
        logging.basicConfig(
            filename=os.path.join(args.output, args.log_file),
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO)
        warnings.filterwarnings("ignore")
        
    if utils.is_main_process():
        logging.info(str(args))

    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train (currently ignored)
    if args.test is not None:
        gqa.predict(
            get_tuple(args.test, bs=args.batch_size), "test"
        )
    else:
        gqa.train(gqa.train_tuple, gqa.valid_tuple)
