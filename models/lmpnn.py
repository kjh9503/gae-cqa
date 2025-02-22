import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy

qtype2edges = {'1p' : {'num_layer' : 1, 'edges' : [[2, 3, 4, 0]], 'position' : [[0, 1, 2]], 'role' : [[0, 3, 2]]},
               '2p' : {'num_layer' : 2, 'edges' : [[2, 3, 4, 0], [4, 5, 6, 0]], 'position' : [[0, 1, 2], [2, 3, 4]],
                       'role' : [[0, 3, 1], [1, 3, 2]]},
               '3p' : {'num_layer' : 3, 'edges' : [[2, 3, 4, 0], [4, 5, 6, 0], [6, 7, 8, 0]],
                       'position' : [[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                       'role' : [[0, 3, 1], [1, 3, 1], [1, 3, 2]]},
               '2i' : {'num_layer' : 1, 'edges' : [[2, 4, 6, 0], [3, 5, 6, 0]],
                       'position' : [[0, 1, 2], [2, 3, 4]],
                       'role' : [[0, 3, 2], [0, 3, 2]]},
               '3i' : {'num_layer' : 1, 'edges' : [[2, 5, 8, 0], [3, 6, 8, 0], [4, 7, 8, 0]],
                       'position' : [[0, 1, 2], [2, 3, 4], [2, 3, 4]],
                       'role' : [[0, 3, 2], [0, 3, 2], [0, 3, 2]]},
               '2in' : {'num_layer' : 1, 'edges' : [[2, 4, 6, 0], [3, 5, 6, 1]],
                        'position' : [[0, 1, 2], [2, 3, 4]],
                        'role' : [[0, 3, 2], [0, 3, 2]]},
               '3in' : {'num_layer' : 1, 'edges' : [[2, 5, 8, 0], [3, 6, 8, 0], [4, 7, 8, 1]],
                        'position' : [[0, 1, 2], [2, 3, 4], [2, 3, 4]],
                        'role' : [[0, 3, 2], [0, 3, 2], [0, 3, 2]]},
                # 'ip' : {'num_layer' : 2, 'edges' : [[2, 4, 6, 0], [3, 5, 6, 0], [6, 7, 8, 0]]},
               'inp' : {'num_layer' : 2, 'edges' : [[2, 4, 6, 0], [3, 5, 6, 1], [6, 7, 8, 0]],
                        'position' : [[0, 1, 2], [2, 3, 4], [2, 3, 4]],
                        'role' : [[0, 3, 1], [0, 3, 1], [1, 3, 2]]},
               # 'pi' : {'num_layer' : 2, 'edges' : [[2, 3, 4, 0], [4, 6, 8, 0], [5, 7, 8, 1]]},
               'pin' : {'num_layer' : 2, 'edges' : [[2, 3, 4, 0], [4, 6, 8, 0], [5, 7, 8, 1]],
                        'position' : [[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                        'role' : [[0, 3, 1], [1, 3, 2], [0, 3, 2]]},
               'pni' : {'num_layer' : 2, 'edges' : [[2, 3, 4, 0], [4, 6, 8, 1], [5, 7, 8, 0]],
                        'position' : [[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                        'role' : [[0, 3, 1], [1, 3, 2], [0, 3, 2]]}
               }
qtype2ent_ind = {'1p' : [2, 4], '2p' : [2, 4, 6], '3p' : [2, 4, 6, 8], '2i' : [2, 3, 6], '3i' : [2, 3, 4, 8],
                  '2in' : [2, 3, 6], '3in' : [2, 3, 4, 8], 'inp' : [2, 3, 6, 8], 'pin' : [2, 4, 5, 8], 'pni' : [2, 4, 5, 8]}
qtype2rel_ind = {'1p' : [3], '2p' : [3, 5], '3p' : [3, 5, 7], '2i' : [4, 5], '3i' : [5, 6, 7],
                 '2in' : [4, 5], '3in' : [5, 6, 7], 'inp' : [4, 5, 7], 'pin' : [3, 6, 7], 'pni' : [3, 6, 7]}

qtype2edges_aug = {}
for qtype in qtype2edges :
    qtype2edges_aug[qtype] = deepcopy(qtype2edges[qtype])
    for ind in qtype2ent_ind[qtype] :
        qtype2edges_aug[qtype]['edges'].append([ind, 10, 0, 0])
    for ind in qtype2rel_ind[qtype] :
        qtype2edges_aug[qtype]['edges'].append([ind, 11, 1, 0])


class LMPNN_Encoder(nn.Module):
    """
    data format [batch, dim]
    """
    def __init__(self, hidden_dim, nbp, layers=1, eps=0.1, agg_func='sum', mp_rel=False, mp_lhs=True):
        super(LMPNN_Encoder, self).__init__()
        self.nbp = nbp
        self.feature_dim = nbp.entity_embedding.size(1)

        self.hidden_dim = hidden_dim
        self.num_entities = nbp.num_entities
        self.agg_func = agg_func

        self.eps = eps

        # ===============================================================
        self.mp_rel = mp_rel
        self.mp_lhs = mp_lhs
        # ===============================================================

        self.existential_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.universal_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.free_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.layer_to_terms_embs_dict = {}
        if layers == 0:
            self.mlp = lambda x: x
        elif layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        elif layers == 2:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        elif layers == 3:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        elif layers == 4:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )

    def message_passing(self, node_embedding, query_type, simple_graph):
        term_collect_embs_dict = defaultdict(list)
        num_layer = qtype2edges[query_type]['num_layer']
        edges = qtype2edges[query_type]['edges']
        for _ in range(num_layer) :
            for edge in edges:
                h_ind, r_ind, t_ind, negated = edge
                if simple_graph :
                    h_ind, r_ind, t_ind = h_ind-2, r_ind-2, t_ind-2
                head_emb = node_embedding[:, h_ind]
                tail_emb = node_embedding[:, t_ind]
                pred_emb = node_embedding[:, r_ind]
                sign = -1 if negated else 1
                term_collect_embs_dict[t_ind].append(
                    sign * self.nbp.estimate_tail_emb(head_emb, pred_emb)
                )
                # ======================================================================
                if self.mp_lhs :
                    term_collect_embs_dict[h_ind].append(
                        sign * self.nbp.estimate_head_emb(tail_emb, pred_emb)
                    )

                # ======================================================================
                if self.mp_rel:
                    term_collect_embs_dict[r_ind].append(
                        sign * self.nbp.estimate_rel_emb(head_emb, tail_emb)
                    )
                # ======================================================================

        return term_collect_embs_dict

    def forward(self, node_embedding, query_type, simple_graph=False):
        term_collect_embs_dict = self.message_passing(node_embedding, query_type, simple_graph)
        if self.agg_func == 'sum':
            term_agg_emb_dict = {
                t: sum(collect_emb_list) + node_embedding[:, t] * self.eps
                for t, collect_emb_list in term_collect_embs_dict.items()
            }
        elif self.agg_func == 'mean':
            term_agg_emb_dict = {
                t: sum(collect_emb_list) / len(collect_emb_list) + node_embedding[:, t] * self.eps
                for t, collect_emb_list in term_collect_embs_dict.items()
            }
        else:
            raise NotImplementedError
        out_term_emb_dict = {
            t: self.mlp(aggemb)
            for t, aggemb in term_agg_emb_dict.items()
        }
        return out_term_emb_dict
    


    
if __name__ == '__main__':
    from . import get_nbp_class
    model_name = 'ComplEx'
    num_entities, num_relations, embedding_dim = 14505, 474, 1000
    device = 'cuda:0'
    bs, seq_len = 4, 10
    hidden_dim = 4096
    mlp_layers=2

    position_embedding = torch.rand((8, 108)).to(device)

    # nbp = ComplEx(num_entities, num_relations, embedding_dim//2, device='cuda')
    nbp = get_nbp_class('ComplEx')(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        device=device)

    nbp.load_state_dict(torch.load('pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt'), strict=True)
    nbp.to(device)
    lgnn_layer = LogicalMPLayer_Q_Strc(hidden_dim, nbp, position_embedding, mlp_layers, eps=0.1, agg_func='sum')
    lgnn_layer.to(device)
    node_embedding = torch.rand((bs, seq_len, 2 * embedding_dim)).to(device)
    query_type = 'pin'
    out_term_emb_dict = lgnn_layer(node_embedding, query_type)
    print(out_term_emb_dict)


