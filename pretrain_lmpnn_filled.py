import itertools
import json
import logging
import pickle
import os

import random
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from args import parse_args
from models import get_nbp_class
from models import QueryGraphAutoencoder

from dataset import SingledirectionalOneShotIterator, query_structure_to_type, load_data, Q2TTrainDataset_Filled, Q2TTestDataset, flatten_query, flatten_query_
from tensorboardX import SummaryWriter

from collections import defaultdict

from utils import set_global_seed, eval_tuple, save_model, log_metrics_qtype, get_mean_val_of_dicts, TYPE_TO_SAMPLE_WEIGHT

import csv


def evaluate(model, tp_answers, fn_answers, args, query_type_to_iterator, query_name_dict, mode, step, writer,
             final_test=False):
    '''
    Evaluate queries in dataloader
    '''
    all_metrics = defaultdict(float)

    metrics = model.test_step(tp_answers, fn_answers, args, query_type_to_iterator, query_name_dict,
                              final_test=final_test)

    p_query_types = [query_type for query_type in metrics.keys() if 'n' not in query_type]
    n_query_types = [query_type for query_type in metrics.keys() if 'n' in query_type]

    for types, name in zip([p_query_types, n_query_types], ['Ap', 'An']):
        average_metrics = defaultdict(float)
        num_query_structures = 0
        num_queries = 0

        for query_type in types:
            # {'1p': {mrr:..., hit1:...,...}, '2p': {...},}
            log_metrics_qtype(mode + " " + query_type, step, metrics[query_type], query_type)
            for metric in metrics[query_type]:
                writer.add_scalar("_".join([mode, query_type, metric]),
                                  metrics[query_type][metric], step)
                all_metrics["_".join([query_type, metric])] = metrics[query_type][metric]
                if metric != 'num_queries':
                    average_metrics[metric] += metrics[query_type][metric]
            num_queries += metrics[query_type]['num_queries']
            num_query_structures += 1

        for metric in average_metrics:
            average_metrics[metric] /= num_query_structures
            writer.add_scalar("_".join([name, mode, 'average', metric]), average_metrics[metric], step)
            all_metrics["_".join(["average", metric])] = average_metrics[metric]
        log_metrics_qtype(f'{mode} {name} average', step, average_metrics)

    return all_metrics


def main(args):
    set_global_seed(args.seed)
    torch.set_num_threads(args.cpu_num)

    tasks, num_ents, num_rels = args.tasks, args.num_ents, args.num_rels

    device = torch.device('cuda' if args.cuda else 'cpu')

    TrainDataset_, TestDataset_ = (Q2TTrainDataset_Filled, Q2TTestDataset)

    if not args.do_train:  # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)

    # construct edge to tail entities
    logging.info(f'constructing s to po')
    df = pd.read_csv('%s/train.txt' % args.data_path, sep='\t', names=['h', 'r', 't'])
    edge_to_entities = defaultdict(list)
    for r, t in zip(df.r.values, df.t.values):
        edge_to_entities[r].append(t)

    logging.info('-------------------------------' * 3)
    logging.info('Geo: %s' % args.geo)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#ent: %d' % num_ents)
    logging.info('#rel: %d' % num_rels)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)

    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data(
        args, tasks)
    # =======================================================================================================================================
    with open(os.path.join(args.data_path, 'var_filled', 'train-queries-filled.pkl'), 'rb') as f:
        train_queries_filled = pickle.load(f)
    # =======================================================================================================================================

    logging.info("Training info:")
    if args.do_train:
        for query_structure in train_queries :
            logging.info(query_structure_to_type[query_structure] + ": " + str(len(train_queries[query_structure])))
        all_train_queries = defaultdict(set)

        for query_structure in train_queries:
            all_train_queries[query_structure] = train_queries[query_structure]

        if args.ignore_1p:
            all_train_queries.pop(('e', ('r',)))

        qtype_train_iterators = []
        qtype_iterators = []
        for query_structure in train_queries:
            qtype = query_structure_to_type[query_structure]
            if qtype in train_queries_filled :
                filled_queries = train_queries_filled[qtype]
            else :
                filled_queries = None

            qtype_train_query = flatten_query_(query_structure, train_queries[query_structure])
            train_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset_(qtype_train_query, num_ents, num_rels, args.negative_sample_size,
                              train_answers, args.data_path, args.enc_dist, filled_queries),
                batch_size=args.batch_size,
                num_workers=args.cpu_num,
                collate_fn=TrainDataset_.collate_fn,
                shuffle=True,
            ))
            qtype_train_iterators.append(train_iterator)
            qtype_iterators.append(query_structure_to_type[query_structure])

    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(query_structure_to_type[query_structure] + ": " + str(len(valid_queries[query_structure])))

        valid_query_type_to_iterator = {}
        for query_structure in valid_queries:
            query_type = query_structure_to_type[query_structure]
            queries = valid_queries[query_structure]
            queries = [(query, query_type) for query in queries]
            valid_query_type_to_iterator[query_type] = DataLoader(
                TestDataset_(
                    queries,
                    args.num_ents,
                    args.num_rels,
                    args.enc_dist
                ),
                batch_size=args.test_batch_size,
                num_workers=args.cpu_num,
                collate_fn=TestDataset_.collate_fn
            )

    logging.info("Test info:")
    if args.do_test:
        for query_structure in test_queries:
            logging.info(query_structure_to_type[query_structure] + ": " + str(len(test_queries[query_structure])))

        test_query_type_to_iterator = {}
        for query_structure in test_queries:
            query_type = query_structure_to_type[query_structure]
            queries = test_queries[query_structure]
            queries = [(query, query_type) for query in queries]
            test_query_type_to_iterator[query_type] = DataLoader(
                TestDataset_(
                    queries,
                    args.num_ents,
                    args.num_rels,
                    args.enc_dist
                ),
                batch_size=args.test_batch_size,
                num_workers=args.cpu_num,
                collate_fn=TestDataset_.collate_fn
            )

    Model = QueryGraphAutoencoder

    if args.ckpt_path is not None:
        with open(os.path.join(args.ckpt_path, 'config.json')) as f:
            kwargs = json.load(f)
    else:
        kwargs = vars(args)

    # ===================================
    nbp = get_nbp_class(args.geo)(
        num_entities=num_ents,
        num_relations=num_rels,
        embedding_dim=args.dim_ent_embedding // 2,
        device=device)

    nbp.load_state_dict(torch.load(args.cqd_ckpt_path), strict=True)
    # ===================================
    model = Model(edge_to_entities, nbp, kwargs)

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        logging.info('moving model to cuda...')
        model.cuda()

    if args.do_train:
        current_learning_rate = args.learning_rate

        # Prepare optimizer
        param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(
            # filter(lambda p: p.requires_grad, model.parameters()),
            optimizer_grouped_parameters,
            lr=current_learning_rate,
        )

        def lr_lambda(current_step: int):
            # 自定义函数
            if current_step < args.warm_up_steps:
                return current_step / max(1, args.warm_up_steps)
            else:
                return 1 - 1.0 * (current_step - args.warm_up_steps) / (args.max_steps - args.warm_up_steps)

        def constant_lambda(step):
            return 1

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if args.ckpt_path is not None:
        logging.info('Loading checkpoint %s...' % args.ckpt_path)
        checkpoint = torch.load(os.path.join(args.ckpt_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0

    step = init_step
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)

    if args.do_train:

        all_structure_to_weights = []

        training_logs = []
        # ============================================================
        best_loss = 100000000
        # ============================================================

        num_query_types = len(qtype_train_iterators)
        for step in range(init_step, (args.max_steps)) :
            # =========================================================================================================
            ind = random.choice(range(num_query_types))
            train_iterator, query_type = qtype_train_iterators[ind], qtype_iterators[ind]
            iter = train_iterator
            # =========================================================================================================

            log = model.pretrain(optimizer, iter, args, task=args.pred)

            for metric in log:
                writer.add_scalar(metric, log[metric], step)

            training_logs.append(log)
            lr_scheduler.step()

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args,
                                                    valid_query_type_to_iterator,
                                                    query_structure_to_type, 'Valid', step, writer)

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args,
                                                test_query_type_to_iterator,
                                                query_structure_to_type, 'Test', step, writer)

                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                }
                save_model(model, optimizer, save_variable_list, args, lr_scheduler)


            loss = log['loss']
            if loss < best_loss:
                best_loss = loss
                if step >= 10000:
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                    }
                    save_model(model, optimizer, save_variable_list, args, lr_scheduler)
                    logging.info('Saved at Step : {}, Loss : {}'.format(step, loss))


            if step % args.log_steps == 0 and step > 0:
                metrics = {}

                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)

                log_metrics_qtype('Training average', step, metrics, query_type)
                training_logs = []



    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_query_type_to_iterator,
                                    query_structure_to_type,
                                    'Test', step, writer, final_test=True)

    logging.info("Training finished!!")

    exit()


if __name__ == '__main__':
    main(parse_args())
