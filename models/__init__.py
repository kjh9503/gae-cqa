# from .query2triple import Query2Triple, AutoEncoder, AutoEncoder_DIST7, AutoEncoder_PERM_DIST, AutoEncoder_DIST
# from .query2triple import AutoEncoder10, AutoEncoder11, AutoEncoder13, AutoEncoder14_3, AutoEncoder15, AutoEncoder15_2
# from .query2triple import AutoEncoder15_3, AutoEncoder16, AutoEncoder16_2, Query2Triple_dist, AutoEncoder_LMPNN
from .query2triple import QueryGraphAutoencoder
from .lmpnn import LMPNN_Encoder


def get_nbp_class(name):
    if name.lower() == 'complex':
        from .nbp_complex import ComplEx
        return ComplEx

