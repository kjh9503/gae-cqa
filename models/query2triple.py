import collections
import logging
import time
import os

import math
import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.cuda.amp import autocast, GradScaler
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch.nn.init import xavier_normal_, kaiming_uniform_, kaiming_normal_
from torch_geometric.data import Data
from transformers import BertConfig
from torch_geometric.nn import MLP
from .modeling_bert import BertModel

from .lmpnn import LMPNN_Encoder

from tqdm import tqdm
from utils import OFFSET, REL_CLS, ENT_CLS, TYPE_TO_IDX, \
    TYPE_TO_SMOOTH, expand, calc_metric_and_record, TGT, VAR, PAD, IDX_TO_TYPE

import torch.nn.functional as F

import random

qtype_to_spc_token_ind = {'1p' : {TGT : 4},
                          '2p' : {TGT : 6, VAR : [4]},
                          '3p' : {TGT : 8, VAR : [4, 6]},
                          '2i' : {TGT : 6},
                          '3i' : {TGT : 8},
                          '2in' : {TGT : 6},
                          '3in' : {TGT : 8},
                          'inp' : {TGT : 6, VAR : [6]},
                          'pni' : {TGT : 8, VAR : [4]},
                          'pin' : {TGT : 8, VAR : [4]}}

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean"):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y, smoothing=None):
        if smoothing is None:
            smoothing = self.smoothing

        return smoothing * x + (1 - smoothing) * y

    def forward(self, preds, target, query_types=None):
        if query_types is not None:
            smoothing = torch.ones(query_types.shape, device=query_types.device)
            for type_ in TYPE_TO_SMOOTH.keys():
                idx = TYPE_TO_IDX[type_]
                smoothing[query_types == idx] = TYPE_TO_SMOOTH[type_]
        else:
            assert 0 <= self.smoothing < 1
            smoothing = self.smoothing

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1)) / n
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction
        )
        return self.linear_combination(loss, nll, smoothing), log_preds

class QueryGraphAutoencoder(nn.Module):
    def __init__(self, edge_to_entities, nbp, kwargs, mask_rate=0.4):
        super(QueryGraphAutoencoder, self).__init__()
        self.edge_to_entities = edge_to_entities
        self.device = torch.device('cuda') if kwargs['cuda'] else torch.device('cpu')
        self.alpha_l = kwargs['alpha_l']
        self.encoder = LMPNN_Encoder(kwargs['mp_hidden_dim'], nbp, kwargs['mp_layers'], kwargs['eps'], kwargs['agg_func'], kwargs['mp_rel'])
        self.decoder = Q2T_Decoder(edge_to_entities = edge_to_entities, **kwargs)
        self.mask_rate = kwargs['mask_rate']
        self.mask_token = nn.Parameter(torch.zeros((self.decoder.dim_ent_embedding,)))
        if kwargs['dec_mask'] == 'dec_mask' :
            self.dec_mask_token = nn.Parameter(torch.zeros((self.decoder.dim_ent_embedding,)))
        if kwargs['pred'] :
            self.loss_fct = LabelSmoothingLoss(smoothing=kwargs['label_smoothing'], reduction='none')
            self.pred_projection = nn.Linear(kwargs['dim_ent_embedding'], kwargs['num_ents'])# -> pretrain1, pretrain2
            if 'ce' in kwargs['pred'] :
                self.loss_fct = nn.CrossEntropyLoss()
        # ========================================================================
        self.spc_tok_mask = kwargs['spc_tok_mask']
        self.mask_cands = kwargs['mask_cands']
        # ========================================================================
            
        nn.init.normal_(self.mask_token, std=0.02)

    def gen_mask(self, batch, cands='node'):
        batch_size, seq_len = batch['x'].size(0), batch['x'].size(1)
        if cands == 'node' :
            seq = batch['x']
        else :
            seq = batch['filled_x']

        mask_cands_r, mask_cands_c = torch.where((seq >= OFFSET) | (seq < 0))
        num_mask = len(mask_cands_r)

        
        selected_mask_inds = torch.LongTensor(np.random.choice(list(range(num_mask)),
                                                               replace=False,
                                                               size=int(num_mask * self.mask_rate))).to(self.device)
        mask_target_inds_r = mask_cands_r[selected_mask_inds]
        mask_target_inds_c = mask_cands_c[selected_mask_inds]

        mask_target_inds = (mask_target_inds_r, mask_target_inds_c)
        return mask_target_inds

    def masking(self, node_embedding, batch, mask='token'):
        mask_inds = self.gen_mask(batch)
        if mask == 'random' :
            mask_token = self.mask_token
        else :
            mask_token = 0
        node_embedding[mask_inds] = mask_token
        return node_embedding, mask_inds

    def forward(self, batch, enc_mask='no', dec_mask='no'):
        mask_inds = []
        # =====================================================================
        filled_x = batch['filled_x']
        # =====================================================================
        query_type = IDX_TO_TYPE[batch['query_type'][0].item()]
        node_embedding = self.decoder.init_seq_embedding(filled_x) #(bs, 10, 2000)
        if enc_mask == 'no':
            pass
        elif enc_mask == 'spc' :
            assert dec_mask == 'no'
            seq = batch['x']
            sp_tok_idx = torch.where((seq >= 0) & (seq < OFFSET))
            node_embedding[sp_tok_idx] = self.decoder.sp_token_embedding(seq[sp_tok_idx])

        else :
            node_embedding, mask_inds = self.masking(node_embedding, batch, mask=enc_mask)
            
        out_term_emb_dict = self.encoder(node_embedding, query_type) # z_ : 768 (4, 10, 2000)

        z = node_embedding.new_zeros(node_embedding.size())

        # ================================================================================
        inds = set(range(filled_x.size(1)))
        for ind in out_term_emb_dict:
            z[:, ind] = out_term_emb_dict[ind]
            inds.remove(ind)
        for _ind in inds :
            z[:, _ind] = node_embedding[:, _ind]
        # ================================================================================

        if dec_mask == 'zero' :
            if enc_mask == 'no' :
                mask_inds = self.gen_mask(batch)
            z[mask_inds] = 0
        elif dec_mask == 'random' :
            if enc_mask == 'no' :
                mask_inds = self.gen_mask(batch)
            z[mask_inds] = self.mask_token
        elif dec_mask == 'dec_mask' :
            if enc_mask == 'no' :
                mask_inds = self.gen_mask(batch)
            z[mask_inds] = self.dec_mask_token
        else :
            pass

        # ==================================================================================================================
        if self.spc_tok_mask :
            seq = batch['x']
            sp_tok_idx = torch.where((seq >= 0) & (seq < OFFSET))
            z[sp_tok_idx] = self.decoder.sp_token_embedding(seq[sp_tok_idx])

        # ==================================================================================================================
        heads_ = self.decoder.forward_hidden(batch, z) # heads_ : 768
        heads = self.decoder.rev_proj(batch, heads_)

        return heads, mask_inds

    def get_feature(self, batch, recall='node'):
        """
        :param seq:
        :param embedding1:
        :param embedding2:
        :return:
        """
        #  init sqe embedding
        seq = batch['filled_x']
    
        shape = seq.shape + (self.decoder.ent_embedding.embedding_dim,)
        node_embeddings = torch.zeros(shape, device=self.device)

        # ent idx
        tup_idx = torch.where(seq >= OFFSET)
        node_embeddings[tup_idx] = self.decoder.ent_embedding(seq[tup_idx] - OFFSET)

        # 0: var, 1: tgt, 2: ENT_CLS, 3: REL_CLS
        # tup_idx = torch.where((seq >= 0) & (seq < OFFSET))
        # node_embeddings[tup_idx] = self.decoder.sp_token_embedding(seq[tup_idx])

        # rel idx
        tup_idx = torch.where(seq < 0)
        node_embeddings[tup_idx] = self.decoder.rel_embedding(torch.abs(seq[tup_idx]) - 1)

        return node_embeddings


    def pretrain(self, optimizer, train_iterator, args, query_type=None, task=None):
        """
        :param self:
        :param optimizer:
        :param train_iterator:
        :param args:
        :return:
        """
        self.train()
        optimizer.zero_grad()

        # freeze bn in KGE
        # for m in self.encoder.kge_model.modules():
        #     if isinstance(m, nn.BatchNorm1d):
        #         m.eval()

        for m in self.decoder.kge_model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()


        t1 = time.time()

        batch = next(train_iterator)

        # t2 = time.time()
        # print('loading ', t2 - t1)
        # t1 = t2

        if args.cuda:
            for k in batch:
                batch[k] = batch[k].cuda()


        with autocast(enabled=self.decoder.fp16, dtype=torch.float16):
            reconstruction_, mask_inds = self.forward(batch, args.enc_mask, args.dec_mask)
            input_feature_ = self.get_feature(batch, recall=args.recon)

            if args.recon == 'mask' :
                reconstruction = reconstruction_[mask_inds]
                input_feature = input_feature_[mask_inds]
            
            # ==========================================================================================
            elif args.recon == 'mask.tgt' :
                recon_mask = reconstruction_[mask_inds]
                input_feat_mask = input_feature_[mask_inds]
                
                tgt_idx = torch.where((batch['x'] == TGT))
                recon_ans = reconstruction_[tgt_idx]
                input_feat_ans = input_feature_[tgt_idx]

                reconstruction = torch.cat([recon_mask, recon_ans], dim=0)
                input_feature = torch.cat([input_feat_mask, input_feat_ans], dim=0)
            
            elif args.recon == 'mask.spc' : #e, r, 만
                recon_mask = reconstruction_[mask_inds]
                input_feat_mask = input_feature_[mask_inds]

                tup_idx = torch.where((batch['x'] == VAR) | (batch['x'] == TGT))
                recon_spc = reconstruction_[tup_idx] 
                input_feat_spc = input_feature_[tup_idx]

                reconstruction = torch.cat([recon_mask, recon_spc], dim=0)
                input_feature = torch.cat([input_feat_mask, input_feat_spc], dim=0)
            
            elif args.recon == 'mask.var' : #e, r, 만
                recon_mask = reconstruction_[mask_inds]
                input_feat_mask = input_feature_[mask_inds]

                tup_idx = torch.where((batch['x'] == VAR))
                recon_spc = reconstruction_[tup_idx] 
                input_feat_spc = input_feature_[tup_idx]

                reconstruction = torch.cat([recon_mask, recon_spc], dim=0)
                input_feature = torch.cat([input_feat_mask, input_feat_spc], dim=0)

            # ==========================================================================================

            elif args.recon == 'node' :
                tup_idx = torch.where((batch['x'] >= OFFSET) | (batch['x'] < 0))
                reconstruction = reconstruction_[tup_idx]
                input_feature = input_feature_[tup_idx]
            elif args.recon == 'all' :
                tup_idx = torch.where((batch['filled_x'] >= OFFSET) | (batch['filled_x'] < 0)) #pad, gh, gr 제외
                reconstruction = reconstruction_[tup_idx]
                input_feature = input_feature_[tup_idx]


            loss_sce  = sce_loss(reconstruction, input_feature, alpha=self.alpha_l)

            loss_sce = loss_sce.mean(-1).mean(-1)

            log = {'sce_loss' : loss_sce.item()}

            if task =='tgt_pred' :
                target = batch.pop('positive_samples')
                tgt_idx = torch.where(batch['x'] == TGT)
                target_logit = reconstruction_[tgt_idx] #(bs, dim)
                target_preds = self.pred_projection(target_logit) #(bs, 14505)

                loss_pred, _ = self.loss_fct(target_preds, target)

                loss_pred = loss_pred.mean() * 0.1

                log.update({'pred_loss': loss_pred.item()})

                loss = loss_sce + loss_pred

                log.update({'loss' : loss.item()})
            
            elif task =='tgt_pred_ns' :
                # target = batch.pop('positive_samples')
                # =================================================
                positive_samples = batch.pop('positive_samples')
                negative_samples = batch.pop('negative_samples')

                tgt_ent_idx = torch.cat([negative_samples, positive_samples.view(-1, 1)], dim=-1) #(bs, 513)
                target = torch.LongTensor([tgt_ent_idx.shape[-1] - 1] * tgt_ent_idx.shape[0]).cuda() # [512, 512, ... ] : bs개
                # =================================================
                tgt_idx = torch.where(batch['x'] == TGT)
                target_logit = reconstruction_[tgt_idx] #(bs, dim)
                # =================================================
                preds = self.decoder.kge_model.get_preds(target_logit, tgt_ent_idx) #(bs, 513)

                # target_preds = self.pred_projection(target_logit) #(bs, 14505)

                loss_pred, _ = self.loss_fct(preds, target)

                loss_pred = loss_pred.mean() * 0.1

                log.update({'pred_loss': loss_pred.item()})

                loss = loss_sce + loss_pred

                log.update({'loss' : loss.item()})

            elif task == 'tgt_pred_ce':
                target = batch.pop('positive_samples')
                tgt_idx = torch.where(batch['x'] == TGT)
                target_logit = reconstruction_[tgt_idx] #(bs, dim)
                target_preds = self.pred_projection(target_logit) #(bs, 14505)

                loss_pred = self.loss_fct(target_preds, target)

                loss_pred = 0.1 * loss_pred

                log.update({'tgt_pred_loss' : loss_pred.item()})

                loss = loss_sce + loss_pred

                log.update({'loss' : loss.item()})

            elif task == 'ghgr_pred' :
                target = batch.pop('positive_samples')
                
                h = reconstruction_[:, 0] #(bs, 2, dim)
                r = reconstruction_[:, 1]
                target_logit = self.decoder.kge_model(h, r) #(bs, dim) - query embedding

                target_preds = self.pred_projection(target_logit)

                loss_pred, _ = self.loss_fct(target_preds, target)

                loss_pred = loss_pred.mean() * 0.1

                log.update({'pred_loss' : loss_pred.item()})

                loss = loss_sce + loss_pred

                log.update({'loss' : loss.item()})

            elif task == 'ghgr_pred_ls' :
                positive_samples = batch.pop('positive_samples')
                negative_samples = batch.pop('negative_samples')

                tgt_ent_idx = torch.cat([negative_samples, positive_samples.view(-1, 1)], dim=-1)
                target = torch.LongTensor([tgt_ent_idx.shape[-1] - 1] * tgt_ent_idx.shape[0]).cuda()

                h = reconstruction_[:, 0] #(bs, 2, dim)
                r = reconstruction_[:, 1]
                t = self.decoder.kge_model(h, r)
                pred = self.decoder.kge_model.get_preds(t, tgt_ent_idx)
                
                loss_preds, log_preds = self.loss_fct(pred, target)

                loss_preds = 0.1 * loss_preds.mean()

                neg_log_preds = log_preds[:, :-1].mean().item()
                pos_log_preds = log_preds[:, -1].mean().item()

                loss = loss_sce + loss_preds

                log.update({'pos_score' : pos_log_preds,
                            'neg_score' : neg_log_preds,
                            'loss' : loss.item()})
            
            elif task == 'no' :
                loss = loss_sce
                log.update({'loss' : loss.item()})
            else :
                raise NotImplementedError('Not implemented task : {}'.format(task))

        if self.decoder.fp16:
            self.encoder.scaler.scale(loss).backward()
            self.encoder.scaler.step(optimizer)
            self.encoder.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return log


    def pred(self, batch, query_type):
        x = batch['x']

        node_embedding = self.decoder.init_seq_embedding(x)

        if query_type in ['1p', '2u-DNF']:
            h = node_embedding[:, 2]
            r = node_embedding[:, 3]
            t = self.decoder.kge_model(h, r)

            pred = self.decoder.kge_model.get_preds(t)

        else :
            dec_out_ = self.decoder.forward_hidden(batch, node_embedding)
            dec_out = self.decoder.rev_proj(batch, dec_out_)

            tgt_idx = torch.where(batch['x'] == TGT)
            target_logit = dec_out[tgt_idx]
            pred = self.pred_projection(target_logit)

        return pred

    def test_step(self, easy_answers, hard_answers, args, query_type_to_iterator, query_name_dict,
                  save_result=False, save_str="", save_empty=False, final_test=False):
        self.eval()

        step = 0
        total_steps = sum([len(iterator) for iterator in query_type_to_iterator.values()])
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for query_type, iterator in query_type_to_iterator.items():
                print(query_type)
                for queries_unflatten, query_types, batch in tqdm(iterator):
                    if args.cuda:
                        for k in batch:
                            batch[k] = batch[k].cuda()

                    if query_types[0] in ['2u-DNF', 'up-DNF']:
                        # [batch, 2, len]
                        origin_x = batch['x']
                        all_preds = []
                        for i in range(2):
                            batch['x'] = origin_x[:, i]
                            pred = self.pred(batch, query_type)
                            all_preds.append(pred)

                        pred = torch.stack(all_preds, dim=1).max(dim=1)[0]
                    else:
                        pred = self.pred(batch, query_type)

                    tmp_logs, tmp_records = calc_metric_and_record(args, easy_answers, hard_answers, pred,
                                                                   queries_unflatten, query_types)
                    for query_structure, res in tmp_logs.items():
                        logs[query_structure].extend(res)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics


class Q2T_Decoder(nn.Module):
    def __init__(self, num_ents, num_rels, hidden_dim, edge_to_entities, mask_rate=0.4,
                 **kwargs):
        super(Q2T_Decoder, self).__init__()

        # basic setting
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.edge_to_entities = edge_to_entities
        self.device = torch.device('cuda') if kwargs['cuda'] else torch.device('cpu')

        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.dim_rel_embedding = kwargs['dim_rel_embedding']


        model = {
            'tucker': TuckER,
            'complex': Complex,
            'cp': CP,
            'rescal': RESCAL,
            'distmult': DistMult
        }[kwargs['geo'].lower()]

        self.kge_model = model(self.num_ents, self.num_rels,
                               self.dim_ent_embedding, self.dim_rel_embedding,
                               kwargs)

        self.kge_ckpt_path = kwargs['kge_ckpt_path']
        if self.kge_ckpt_path:
            self.kge_model.load_from_ckpt_path(self.kge_ckpt_path)

        self.ent_embedding = self.kge_model.ent_embedding
        self.rel_embedding = self.kge_model.rel_embedding
        kge_requires_grad = True if kwargs['not_freeze_kge'] else False

        if not kge_requires_grad:
            for name, param in self.kge_model.named_parameters():
                param.requires_grad = False

        logging.info(f'KGE requires_grad: {kge_requires_grad}')

        if self.kge_ckpt_path and not kge_requires_grad:
            self.use_kge_to_pred_1p = True
        else:
            self.use_kge_to_pred_1p = False
        logging.info(f'use_kge_to_pred_1p: {self.use_kge_to_pred_1p}')

        # var, tgt, CLS
        self.sp_token_embedding = nn.Embedding(OFFSET, self.dim_ent_embedding)
        # prompt
        self.query_encoder = BertEncoder(kwargs)
        
        # ================================================================================
        self.node_rev_proj = nn.Linear(self.query_encoder.hidden_size, self.dim_ent_embedding)
        # ================================================================================

        self.fp16 = kwargs['fp16']
        if self.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.loss_fct = LabelSmoothingLoss(smoothing=kwargs['label_smoothing'], reduction='none')

        self.init_weight()

    def init_weight(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    # ==============================================================================================================
    def rev_proj(self, batch, embedding):


        proj_embedding = embedding.new_zeros((embedding.size(0), embedding.size(1), self.dim_ent_embedding))

        proj_embedding[:, 0] = self.query_encoder.rev_proj1(embedding[:, 0])
        proj_embedding[:, 1] = self.query_encoder.rev_proj2(embedding[:, 1])
        
        proj_embedding[:, 2:] = self.node_rev_proj(embedding[:, 2:])

        return proj_embedding
    # ==============================================================================================================
    def init_seq_embedding(self, seq: torch.tensor):
        """
        :param seq:
        :param embedding1:
        :param embedding2:
        :return:
        """
        #  init sqe embedding
        shape = seq.shape + (self.ent_embedding.embedding_dim,)
        node_embeddings = torch.zeros(shape, device=self.device)

        # ent idx
        tup_idx = torch.where(seq >= OFFSET)
        node_embeddings[tup_idx] = self.ent_embedding(seq[tup_idx] - OFFSET)

        # 0: pad, 1: var, 2: tgt, 3: ent, 4 : rel
        tup_idx = torch.where((seq >= 0) & (seq < OFFSET))
        node_embeddings[tup_idx] = self.sp_token_embedding(seq[tup_idx])

        # rel idx
        tup_idx = torch.where(seq < 0)
        node_embeddings[tup_idx] = self.rel_embedding(torch.abs(seq[tup_idx]) - 1)

        return node_embeddings

    def forward_hidden(self, batch: Data, x=None, tgt_ent_idx=None):
        # ==============================================================================================================
        if x is None :
            x = batch['x']
            node_embedding = self.init_seq_embedding(x)
        else :
            node_embedding = x

        hidden_states = self.query_encoder(node_embedding, graph=batch, return_hidden=True)
        return hidden_states

    def forward(self, batch: Data, tgt_ent_idx=None):
        """
        """
        x = batch['x']
        node_embedding = self.init_seq_embedding(x)
        hidden_states = self.query_encoder(node_embedding, graph=batch)

        cls1 = hidden_states[:, 0]
        cls2 = hidden_states[:, 1]

        h = self.query_encoder.rev_proj1(cls1)
        r = self.query_encoder.rev_proj2(cls2)
        
        t = self.kge_model(h, r)
        pred = self.kge_model.get_preds(t, tgt_ent_idx)

        return pred, None

    def pred(self, batch, query_type):
        x = batch['x']
        node_embedding = self.init_seq_embedding(x)

        if query_type in ['1p', '2u-DNF'] and self.use_kge_to_pred_1p:
            # test 1p
            h = node_embedding[:, 2]
            r = node_embedding[:, 3]
            t = self.kge_model(h, r)

            pred = self.kge_model.get_preds(t)

        else:
            # ==========================================================================================================
            hidden_states = self.query_encoder(node_embedding, graph=batch) #(bs, 10, 2000)

            cls1 = hidden_states[:, 0]
            cls2 = hidden_states[:, 1]

            h = self.query_encoder.rev_proj1(cls1) #(bs, 2000)
            r = self.query_encoder.rev_proj2(cls2) #(bs, 2000)
            

            t = self.kge_model(h, r)
            pred = self.kge_model.get_preds(t)

        return pred

    def train_step(self, optimizer, train_iterator, args, query_type=None):
        """

        :param self:
        :param optimizer:
        :param train_iterator:
        :param args:
        :return:
        """
        self.train()
        optimizer.zero_grad()

        # freeze bn in KGE
        for m in self.kge_model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        t1 = time.time()

        batch = next(train_iterator)

        # t2 = time.time()
        # print('loading ', t2 - t1)
        # t1 = t2

        if args.cuda:
            for k in batch:
                batch[k] = batch[k].cuda()

        with autocast(enabled=self.fp16, dtype=torch.float16):
            negative_samples = batch.pop('negative_samples')
            positive_samples = batch.pop('positive_samples')

            tgt_ent_idx = torch.cat([negative_samples, positive_samples.view(-1, 1)], dim=-1) #(bs, 512+1)
            target = torch.LongTensor([tgt_ent_idx.shape[-1] - 1] * tgt_ent_idx.shape[0]).cuda() #([512, 512, ... ] :len:bs)

            # tgt_ent_idx = None
            # target = positive_samples

            preds, ws = self.forward(batch, tgt_ent_idx=tgt_ent_idx)

            loss, log_preds = self.loss_fct(preds, target) #preds : (bs, 513), target : (bs,)
            neg_log_preds = log_preds[:, :-1].mean().item()
            pos_log_preds = log_preds[:, -1].mean().item()

            loss = loss.mean()

            log = {
                'pos_score': pos_log_preds,
                'neg_score': neg_log_preds,
                'loss': loss.item()
            }

        if self.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return log

    def test_step(self, easy_answers, hard_answers, args, query_type_to_iterator, query_name_dict,
                  save_result=False, save_str="", save_empty=False, final_test=False):
        self.eval()

        step = 0
        total_steps = sum([len(iterator) for iterator in query_type_to_iterator.values()])
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for query_type, iterator in query_type_to_iterator.items():
                for queries_unflatten, query_types, batch in tqdm(iterator):
                    if args.cuda:
                        for k in batch:
                            batch[k] = batch[k].cuda()

                    if query_types[0] in ['2u-DNF', 'up-DNF']:
                        # [batch, 2, len]
                        origin_x = batch['x']
                        all_preds = []
                        for i in range(2):
                            batch['x'] = origin_x[:, i]
                            pred = self.pred(batch, query_type)
                            all_preds.append(pred)

                        pred = torch.stack(all_preds, dim=1).max(dim=1)[0]
                    else:
                        pred = self.pred(batch, query_type)

                    tmp_logs, tmp_records = calc_metric_and_record(args, easy_answers, hard_answers, pred,
                                                                   queries_unflatten, query_types)
                    for query_structure, res in tmp_logs.items():
                        logs[query_structure].extend(res)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics


class Query2Triple(nn.Module):
    def __init__(self, num_ents, num_rels, hidden_dim, edge_to_entities,
                 **kwargs):
        super(Query2Triple, self).__init__()

        # basic setting
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.edge_to_entities = edge_to_entities
        self.device = torch.device('cuda') if kwargs['cuda'] else torch.device('cpu')

        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.dim_rel_embedding = kwargs['dim_rel_embedding']

        model = {
            'tucker': TuckER,
            'complex': Complex,
            'cp': CP,
            'rescal': RESCAL,
            'distmult': DistMult
        }[kwargs['geo'].lower()]
        self.kge_model = model(self.num_ents, self.num_rels,
                               self.dim_ent_embedding, self.dim_rel_embedding,
                               kwargs)

        self.kge_ckpt_path = kwargs['kge_ckpt_path']
        if self.kge_ckpt_path:
            self.kge_model.load_from_ckpt_path(self.kge_ckpt_path)

        self.ent_embedding = self.kge_model.ent_embedding
        self.rel_embedding = self.kge_model.rel_embedding
        kge_requires_grad = True if kwargs['not_freeze_kge'] else False

        if not kge_requires_grad:
            for name, param in self.kge_model.named_parameters():
                param.requires_grad = False

        logging.info(f'KGE requires_grad: {kge_requires_grad}')

        if self.kge_ckpt_path and not kge_requires_grad:
            self.use_kge_to_pred_1p = True
        else:
            self.use_kge_to_pred_1p = False
        logging.info(f'use_kge_to_pred_1p: {self.use_kge_to_pred_1p}')

        # var, tgt, CLS
        self.sp_token_embedding = nn.Embedding(OFFSET, self.dim_ent_embedding)
        # prompt
        self.query_encoder = BertEncoder(kwargs)

        self.fp16 = kwargs['fp16']
        if self.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.loss_fct = LabelSmoothingLoss(smoothing=kwargs['label_smoothing'], reduction='none')

        self.init_weight()

    def init_weight(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_seq_embedding(self, seq: torch.tensor):
        """
        :param seq:
        :param embedding1:
        :param embedding2:
        :return:
        """
        #  init sqe embedding
        shape = seq.shape + (self.ent_embedding.embedding_dim,)
        node_embeddings = torch.zeros(shape, device=self.device)

        # ent idx
        tup_idx = torch.where(seq >= OFFSET)
        node_embeddings[tup_idx] = self.ent_embedding(seq[tup_idx] - OFFSET)

        # 0: var, 1: tgt, 2: ENT_CLS, 3: REL_CLS
        tup_idx = torch.where((seq >= 0) & (seq < OFFSET))
        node_embeddings[tup_idx] = self.sp_token_embedding(seq[tup_idx])

        # rel idx
        tup_idx = torch.where(seq < 0)
        node_embeddings[tup_idx] = self.rel_embedding(torch.abs(seq[tup_idx]) - 1)

        return node_embeddings

    def forward(self, batch: Data, tgt_ent_idx=None):
        """
        """
        x = batch['x']

        node_embedding = self.init_seq_embedding(x)
        h, r, ws = self.query_encoder(node_embedding, graph=batch)

        t = self.kge_model(h, r)
        pred = self.kge_model.get_preds(t, tgt_ent_idx)

        return pred, ws

    def pred(self, batch, query_type):
        x = batch['x']
        node_embedding = self.init_seq_embedding(x)

        if query_type in ['1p', '2u-DNF'] and self.use_kge_to_pred_1p:
            # test 1p
            h = node_embedding[:, 2]
            r = node_embedding[:, 3]
            t = self.kge_model(h, r)

            pred = self.kge_model.get_preds(t)

        else:
            h, r, _ = self.query_encoder(node_embedding, graph=batch)
            t = self.kge_model(h, r)
            pred = self.kge_model.get_preds(t)

        return pred

    def train_step(self, optimizer, train_iterator, args, query_type=None):
        """

        :param self:
        :param optimizer:
        :param train_iterator:
        :param args:
        :return:
        """
        self.train()
        optimizer.zero_grad()

        # freeze bn in KGE
        for m in self.kge_model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        t1 = time.time()

        batch = next(train_iterator)

        # t2 = time.time()
        # print('loading ', t2 - t1)
        # t1 = t2

        if args.cuda:
            for k in batch:
                batch[k] = batch[k].cuda()

        with autocast(enabled=self.fp16, dtype=torch.float16):
            negative_samples = batch.pop('negative_samples')
            positive_samples = batch.pop('positive_samples')

            tgt_ent_idx = torch.cat([negative_samples, positive_samples.view(-1, 1)], dim=-1)
            target = torch.LongTensor([tgt_ent_idx.shape[-1] - 1] * tgt_ent_idx.shape[0]).cuda()

            # tgt_ent_idx = None
            # target = positive_samples

            preds, ws = self(batch, tgt_ent_idx=tgt_ent_idx)

            loss, log_preds = self.loss_fct(preds, target)
            neg_log_preds = log_preds[:, :-1].mean().item()
            pos_log_preds = log_preds[:, -1].mean().item()

            loss = loss.mean()

            log = {
                'pos_score': pos_log_preds,
                'neg_score': neg_log_preds,
                'loss': loss.item()
            }

        if self.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return log

    def test_step(self, easy_answers, hard_answers, args, query_type_to_iterator, query_name_dict,
                  save_result=False, save_str="", save_empty=False, final_test=False):
        self.eval()

        step = 0
        total_steps = sum([len(iterator) for iterator in query_type_to_iterator.values()])
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for query_type, iterator in query_type_to_iterator.items():
                for queries_unflatten, query_types, batch in tqdm(iterator):
                    if args.cuda:
                        for k in batch:
                            batch[k] = batch[k].cuda()

                    if query_types[0] in ['2u-DNF', 'up-DNF']:
                        # [batch, 2, len]
                        origin_x = batch['x']
                        all_preds = []
                        for i in range(2):
                            batch['x'] = origin_x[:, i]
                            pred = self.pred(batch, query_type)
                            all_preds.append(pred)

                        pred = torch.stack(all_preds, dim=1).max(dim=1)[0]
                    else:
                        pred = self.pred(batch, query_type)

                    tmp_logs, tmp_records = calc_metric_and_record(args, easy_answers, hard_answers, pred,
                                                                   queries_unflatten, query_types)
                    for query_structure, res in tmp_logs.items():
                        logs[query_structure].extend(res)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics

class TokenEmbedding(nn.Module):
    def __init__(self, kwargs):
        super(TokenEmbedding, self).__init__()
        self.kwargs = kwargs
        # [1,2,3]
        if len(kwargs['token_embeddings']):
            self.token_embeds = [int(_) for _ in kwargs['token_embeddings'].split('.')]
        else:
            self.token_embeds = []
        self.hidden_size = kwargs['hidden_size']
        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.p_dropout = kwargs['hidden_dropout_prob']

        self.type_embeddings = nn.Embedding(2, self.hidden_size) if 1 in self.token_embeds else None
        self.layer_embeddings = nn.Embedding(8, self.hidden_size) if 2 in self.token_embeds else None
        self.op_embeddings = nn.Embedding(2, self.hidden_size) if 3 in self.token_embeds else None
        self.in_embeddings = nn.Embedding(8, self.hidden_size) if 4 in self.token_embeds else None
        self.out_embeddings = nn.Embedding(8, self.hidden_size) if 5 in self.token_embeds else None

        self.proj = nn.Linear(self.dim_ent_embedding, self.hidden_size)

        self.n_neg_proj = 1
        self.neg_proj = nn.ModuleList(
            [MLP(channel_list=[self.hidden_size, self.hidden_size])
             for _ in range(self.n_neg_proj)]
        )

        self.norm = nn.LayerNorm(self.hidden_size, eps=kwargs['layer_norm_eps'])
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, node_embeddings, graph):
        node_embeddings = self.proj(node_embeddings)

        if self.type_embeddings:
            node_embeddings += self.type_embeddings(graph['node_types'])

        if self.layer_embeddings:
            node_embeddings += self.layer_embeddings(graph['layers'])

        if self.op_embeddings:
            node_embeddings += self.op_embeddings(graph['operators'])

        if self.in_embeddings:
            node_embeddings += self.in_embeddings(graph['in_degs'])

        if self.out_embeddings:
            node_embeddings += self.out_embeddings(graph['out_degs'])

        for i in range(self.n_neg_proj):
            idxes = torch.where(graph['negs'] == i + 1)
            node_embeddings[idxes] = self.neg_proj[i](node_embeddings[idxes])

        node_embeddings = self.norm(node_embeddings)
        node_embeddings = self.dropout(node_embeddings)

        return node_embeddings


class BertEncoder(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.dim_rel_embedding = kwargs['dim_rel_embedding']
        self.hidden_size = kwargs['hidden_size']
        self.num_heads = kwargs['num_attention_heads']
        self.head_dim = kwargs['hidden_size'] // self.num_heads
        self.device = torch.device('cuda') if kwargs['cuda'] else torch.device('cpu')

        self.embedding = TokenEmbedding(kwargs)

        config = BertConfig(
            num_hidden_layers=kwargs['num_hidden_layers'],
            hidden_size=kwargs['hidden_size'],
            num_attention_heads=kwargs['num_attention_heads'],
            intermediate_size=kwargs['intermediate_size'],
            hidden_dropout_prob=kwargs['hidden_dropout_prob'],
            attention_probs_dropout_prob=kwargs['hidden_dropout_prob'],
            fp16=kwargs['fp16'],
            enc_dist=(kwargs['enc_dist'])
        )
        self.bert = BertModel(config)

        self.rev_proj1 = nn.Linear(self.hidden_size, self.dim_ent_embedding)
        self.rev_proj2 = nn.Linear(self.hidden_size, self.dim_rel_embedding)

    def forward(self, initial_node_embeddings,
                graph, return_hidden=True):
        # [b, l, dim]
        node_embeddings = self.embedding(initial_node_embeddings,
                                         graph)
        # node_embeddings = self.norm(node_embeddings)

        batch, length, dim = node_embeddings.shape

        hidden_states = self.bert(
            inputs_embeds=node_embeddings,
            attention_mask=graph['attention_mask'],
            dist_mat=graph['dist_mat'],
            negs=graph['negs']
        ).last_hidden_state
        
        #==========================================================================================================
        if return_hidden : 
            return hidden_states
        else :
        
            cls1 = hidden_states[:, 0]
            cls2 = hidden_states[:, 1]
            # tgts = hidden_states[torch.where(graph['targets'] == 1)]
            
            h = self.rev_proj1(cls1)
            r = self.rev_proj2(cls2)
        #==========================================================================================================
            return h, r, None

class KGE(nn.Module):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__()
        self.ent_embedding = None
        self.rel_embedding = None

        self.dim_rel_embedding = dim_rel_embedding
        self.dim_ent_embedding = dim_ent_embedding
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.kwargs = kwargs
        self.init_size = 1e-3

    def forward(self, lhs, rel):
        raise NotImplemented

    def load_from_ckpt_path(self, ckpt_path):
        raise NotImplemented

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        raise NotImplemented

    @staticmethod
    def calc_preds(pred_embedding, ent_embedding, tgt_ent_idx=None):
        if tgt_ent_idx is None:
            # [dim, n_ent]
            tgt_ent_embedding = ent_embedding.weight.transpose(0, 1)

            # [n_batch, n_ent]
            scores = pred_embedding @ tgt_ent_embedding

        else:
            # [n_batch, neg, dim]
            tgt_ent_embedding = ent_embedding(tgt_ent_idx)

            # [n_batch, dim, 1]
            pred_embedding = pred_embedding.unsqueeze(-1)

            scores = torch.bmm(tgt_ent_embedding, pred_embedding)
            scores = scores.squeeze(-1)

        return scores


class Complex(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)
        # embedding and W
        self.rank = dim_ent_embedding // 2

        self.ent_embedding = nn.Embedding(num_ents, 2 * self.rank)
        self.rel_embedding = nn.Embedding(num_rels, 2 * self.rank)

        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size

        self.embeddings = [self.ent_embedding, self.rel_embedding]

    def forward_emb(self, lhs, rel, to_score_idx=None):
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        if not to_score_idx:
            to_score = self.embeddings[0].weight
        else:
            to_score = self.embeddings[0](to_score_idx)

        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))

    def forward(self, lhs, rel):
        lhs = torch.chunk(lhs, 2, -1)
        rel = torch.chunk(rel, 2, -1)

        output = ([lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]])
        output = torch.cat(output, dim=-1)

        return output

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def get_factor(self, x):
        lhs = self.ent_embedding(x[0])
        rel = self.rel_embedding(x[1])
        rhs = self.ent_embedding(x[2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        return (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading Complex params from {ckpt_path}')

        try:
            self.embeddings[0].weight.data = params['embeddings.0.weight']
            self.embeddings[1].weight.data = params['embeddings.1.weight']
        except:
            self.embeddings[0].weight.data = params['_entity_embedding.weight']
            self.embeddings[1].weight.data = params['_relation_embedding.weight']

        self.ent_embedding_norm_mean = self.embeddings[0].weight.data.norm(p=2, dim=1).mean().item()
        self.rel_embedding_norm_mean = self.embeddings[1].weight.data.norm(p=2, dim=1).mean().item()

        self.embeddings[0].weight.data /= self.ent_embedding_norm_mean
        self.embeddings[1].weight.data /= self.rel_embedding_norm_mean


class TuckER(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        self.E = torch.nn.Embedding(num_ents, dim_ent_embedding)
        self.R = torch.nn.Embedding(num_rels, dim_rel_embedding)
        self.W = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (dim_rel_embedding, dim_ent_embedding, dim_ent_embedding)),
                         dtype=torch.float))

        self.input_dropout = torch.nn.Dropout(0.3)
        self.hidden_dropout1 = torch.nn.Dropout(0.4)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)

        self.bn0 = torch.nn.BatchNorm1d(dim_ent_embedding)
        self.bn1 = torch.nn.BatchNorm1d(dim_ent_embedding)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, lhs, rel):
        x = self.bn0(lhs)
        x = self.input_dropout(x)
        x = x.view(-1, 1, self.dim_ent_embedding)

        W_mat = torch.mm(rel, self.W.view(self.dim_rel_embedding, -1))
        W_mat = W_mat.view(-1, self.dim_ent_embedding, self.dim_ent_embedding)
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, self.dim_ent_embedding)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        return x

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.E, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))

        self.ent_embedding = self.E
        self.rel_embedding = self.R


class CP(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        self.ent_embedding = nn.Embedding(num_ents, self.dim_ent_embedding)
        self.rel_embedding = nn.Embedding(num_rels, self.dim_rel_embedding)
        self.ent_embedding1 = nn.Embedding(num_ents, self.dim_ent_embedding)

        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size
        self.ent_embedding1.weight.data *= self.init_size

    def forward(self, lhs, rel):
        return lhs * rel

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding1, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading CP params from {ckpt_path}')

        self.ent_embedding.weight.data = params['lhs.weight']
        self.rel_embedding.weight.data = params['rel.weight']
        self.ent_embedding1.weight.data = params['rhs.weight']


class DistMult(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        self.ent_embedding = nn.Embedding(num_ents, self.dim_ent_embedding)
        self.rel_embedding = nn.Embedding(num_rels, self.dim_rel_embedding)

        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size

    def forward(self, lhs, rel):
        return lhs * rel

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading DistMult params from {ckpt_path}')

        self.ent_embedding.weight.data = params['entity.weight']
        self.rel_embedding.weight.data = params['relation.weight']


class RESCAL(KGE):
    def __init__(self, num_ents, num_rels, dim_ent_embedding, dim_rel_embedding, kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        assert dim_rel_embedding == dim_ent_embedding
        self.rank = dim_ent_embedding

        self.ent_embedding = nn.Embedding(num_ents, self.rank)
        self.rel_embedding = nn.Embedding(num_rels, self.rank * self.rank)
        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size

    def forward(self, lhs, rel):
        rel = rel.view(-1, self.rank, self.rank)
        lhs_proj = lhs.view(-1, 1, self.rank)
        lhs_proj = torch.bmm(lhs_proj, rel).view(-1, self.rank)
        return lhs_proj

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading RESCAL params from {ckpt_path}')

        self.ent_embedding.weight.data = params['entity.weight']
        self.rel_embedding.weight.data = params['relation.weight']
