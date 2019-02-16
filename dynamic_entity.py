import tensorflow as tf

class DynamicEntity (object):
    def __init__  (self, name, type, label, base_embedding):
        """
        :param name: name of entity
        :param type: type of the entity. In the paper it was mentioned as r.
        :param label: label of the entity. It can be considered as index of the entity in the corpus.
        :param base_embedding: used for initializing the state of the entity (take a look at the paper :) ).
        """

        raise NotImplementedError


    def update (self, h_t, W_h, W_delta):
        """
        desc. : updates entity according to the hidden state which it receives.

        :param h_t: hidden state of LSTM in EnGen at time step t.
        :param W_h: parameter used to convert h_t to a vector with dimension of state of the entity.
        :param W_delta: used for updating the state of the entity (for more information, take a look at the paper).
        :return: updated state of the entity.
        """

        raise NotImplementedError