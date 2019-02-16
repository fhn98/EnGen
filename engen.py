import tensorflow as tf

class EnGen (tf.keras.Model):
    def __init__ (self, name, embedding_table, max_mention_length, types_num, hidden_dim, entity_dim,
                  pooling_type, compostion_method, initializer, dropout_rate, train):
        """
        :param name: name of the model.
        :param embedding_table: used for embedding_lookup
        :param max_mention_length: max length of a mention. In the paper it was mentioned as l.
        :param types_num: number of types of entity. in the paper there were only entity type and non-entity type.
        :param hidden_dim: hidden dimension of the LSTM cell.
        :param entity_dim: dimension of the entity state.
        :param pooling_type: pooling used for creating context.
        :param compostion_method: method for combining the hidden state,
                                  previous sentence embedding and entity embedding to
                                  create the context (in the paper only one method was mentioned).
        :param initializer: initializer function for parameters.
        :param dropout_rate: dropout rate.
        :param train: if True, it will be trained as a language model, else it will generate sequence from the previous
                      context.
        """

        raise NotImplementedError

    def make_params(self , initializer, hidden_dim, entity_dim, max_mention_length):
        """
        desc. : makes parameters of the model with specified initializer
        """

        raise NotImplementedError

    def create_entity(self, entity_rep, entity_embds, entity_dist, etoi, current_index, sent_num):
        """
        desc. : creates an entity and updates entity_list, entity_dist, itoe, etoi
        :param entity_rep: tensor indicating base embedding for each entity type
        :param entity_embds: tensor containing embedding of entities created so far
        :param entity_dist: tensor containing number of the sentence of the last mention for each entity
        :param etoi: map from entity index to index of the entity in entity_embds (?!*)
        :param current_index: index of the current entity
        :param sent_num: number of the sentence in which entity is created
        :return: the created entity :)
        """

        raise NotImplementedError

    def update_entity(self, entity_embds, entity_dist, etoi, current_index, sent_num):
        """
        desc. : updates entity with index of current_index
        :param entity_embds: tensor containing embedding of entities created so far
        :param entity_dist: tensor containing number of the sentence of the last mention for each entity
        :param etoi: map from entity index to index of the entity in entity_embds (?!*)
        :param current_index: index of the current entity
        :param sent_num: number of the sentence in which entity is created
        """

        raise NotImplementedError

    def get_dist_features (self, entity_dist, sent_num):
        """
        desc. :returns distance feature vector used for prdicting the next entity
        :param entity_dist: tensor containing number of the sentence of the last mention for each entity
        :param sent_num: number of the current sentence
        :return: distance feature vector
        """

        raise NotImplementedError

    def get_context (self, hidden_state, prev_sent, curr_entity, composition_method, pooling_type):
        """
        desc. : creates context from LSTM hidden state, current entity embedding and previous sentence encoding
        :param hidden_state: last hidden state of the LSTM
        :param prev_sent: encoding of the previous sentence
        :param curr_entity: embedding of te current entity
        :param composition_method: method for combining the hidden state,
                                   previous sentence embedding and entity embedding to
                                   create the context (in the paper only one method was mentioned).
        :param pooling_type: pooling used for creating context.
        :return: context created using the input.
        """

        raise NotImplementedError

    def attention_encoder(self, prev_hiddens, curr_hidden):
        """
        desc. : returns an encoding of the previous sentence with attention
        :param prev_hiddens: hidden state of LSTM for each token in the previous sentence
        :param curr_hidden: hidden state of LSTM for the current token
        :return: the encoding which was computed using attention.
        """

        raise NotImplementedError

    def generate (self, prev_hiddens, entity_rep, entity_embds, entity_dist, etoi, itoe,
                  sent_num, composition_method, pooling_type):
        """
            desc. : generates sequence using the inputs :)
        """
        raise NotImplementedError

    def make_prev_context (self, inputs):
        """
            desc: reads the inputs and updates entity states, this method is used before the generate method
        """
        raise NotImplementedError

    def call (self , inputs):
        """
        desc. : if train is True, gives a softmax distribution for predicting the next entity type, entity label,
                mention length, and word.
                else, it generates sequences continuing inputs till it reaches <EOS> token (using two methods:
                make_prev_context and generate)

        :param inputs: if train is True:
                           inputs[0]: [batch_size, max_seq_length] tensor containing entity type
                           inputs[1]: [batch_size, max_seq_length] tensor containing entity label
                           inputs[2]: [batch_size, max_seq_length] tensor containing mention length
                           inputs[3]: [batch_size, max_seq_length] tensor containing index of each token in
                                      embedding table
                       else:
                             [batch_size, max_seq_length] tensor containing index of each token in the context in
                                      embedding table


        :return: if train is True:
                    prediction for the next token entity type, entity label,
                    mention length, and index.
                 else: the generated sequence
        """

        raise NotImplementedError

