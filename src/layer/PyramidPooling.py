from keras.engine.topology import Layer
import keras.backend as K


class PyramidPooling(Layer):
    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i for i in pool_list])

        super(PyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        #if self.dim_ordering == 'th':
            # TODO
        if self.dim_ordering == 'tf':
            self.nb_channels = input_shape[2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(PyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        #if self.dim_ordering == 'th':
            # TODO
        if self.dim_ordering == 'tf':
            num_rows = 1
            num_cols = input_shape[1]

        row_length = [K.cast(num_rows, 'float32') / (i / i) for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        #if self.dim_ordering == 'th':
            # TODO

        if self.dim_ordering == 'tf':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for ix in range(num_pool_regions):
                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]

                    x1 = K.cast(K.round(x1), 'int32')
                    x2 = K.cast(K.round(x2), 'int32')

                    new_shape = [input_shape[0], x2 - x1, input_shape[2]]

                    x_crop = x[:, x1:x2, :]
                    xm = K.reshape(x_crop, new_shape)
                    pooled_val = K.max(xm, axis=(1))
                    outputs.append(pooled_val)

        #if self.dim_ordering == 'th':
            # TODO
        if self.dim_ordering == 'tf':
            outputs = K.concatenate(outputs)

        return outputs
