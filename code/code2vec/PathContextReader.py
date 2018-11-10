import tensorflow as tf
import common

no_such_word = 0
no_such_composite = '{} {} {}'.format(no_such_word, no_such_word, no_such_word)


class PathContextReader:
    class_word_table = None
    class_target_word_table = None
    class_path_table = None

    def __init__(self, config, is_evaluating=False):
        self.file_path = config.TEST_PATH if is_evaluating else config.TRAIN_PATH
        self.batch_size = min(config.TEST_BATCH_SIZE if is_evaluating else config.BATCH_SIZE, config.NUM_EXAMPLES)
        self.num_epochs = config.NUM_EPOCHS
        self.reading_batch_size = min(config.READING_BATCH_SIZE, config.NUM_EXAMPLES)
        self.num_batching_threads = config.NUM_BATCHING_THREADS
        self.batch_queue_size = config.BATCH_QUEUE_SIZE
        self.data_num_contexts = config.MAX_CONTEXTS
        self.data_path_limit = config.PATH_LIMIT
        self.max_contexts = config.MAX_CONTEXTS
        self.is_evaluating = is_evaluating
        self.filtered_output = self.get_filtered_input()

    def get_input_placeholder(self):
        return self.input_placeholder

    def start(self, session, data_lines=None):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=session, coord=self.coord)
        return self

    def read_file(self):
        row = self.get_row_input()
        record_defaults = [[no_such_composite]] * (self.data_path_limit + 1)
        row_parts = tf.decode_csv(row, record_defaults=record_defaults, field_delim=',')
        entities = tf.string_to_number(row_parts[0], out_type=tf.int32)  # (batch, )
        contexts = tf.stack(row_parts[1:(self.max_contexts + 1)], axis=1)  # (batch, max_contexts)

        flat_contexts = tf.reshape(contexts, [-1])  # (batch * max_contexts, )
        split_contexts = tf.string_split(flat_contexts, delimiter=' ')
        dense_split_contexts = tf.reshape(tf.sparse_tensor_to_dense(split_contexts,
                                                                    default_value=str(no_such_word)),
                                          shape=[-1, self.max_contexts, 3])  # (batch, max_contexts, 3)

        dense_split_contexts = tf.string_to_number(dense_split_contexts, out_type=tf.int32)
        start_terminals = tf.slice(dense_split_contexts, [0, 0, 0], [-1, self.max_contexts, 1])
        paths = tf.slice(dense_split_contexts, [0, 0, 1], [-1, self.max_contexts, 1])
        end_terminals = tf.slice(dense_split_contexts, [0, 0, 2], [-1, self.max_contexts, 1])

        return entities, start_terminals, paths, end_terminals

    def get_row_input(self):
        if self.is_evaluating:  # test, read from queue (small data)
            row = self.input_placeholder = tf.placeholder(tf.string)
        else:  # training, read from file
            filename_queue = tf.train.string_input_producer([self.file_path], num_epochs=self.num_epochs, shuffle=False)
            reader = tf.TextLineReader()
            _, row = reader.read_up_to(filename_queue, num_records=self.reading_batch_size)
        return row

    def input_tensors(self):
        return self.initialize_batch_outputs(self.filtered_output)

    def get_filtered_batches(self):
        return self.filtered_output

    def initialize_batch_outputs(self, filtered_input):
        return tf.train.shuffle_batch(filtered_input,
                                      batch_size=self.batch_size,
                                      enqueue_many=True,
                                      capacity=self.batch_queue_size,
                                      min_after_dequeue=int(self.batch_queue_size * 0.85),
                                      num_threads=self.num_batching_threads,
                                      allow_smaller_final_batch=True)

    def get_filtered_input(self):
        entities, start_terminals, paths, end_terminals = self.read_file()
        any_contexts_is_valid = tf.logical_or(
            tf.greater(tf.squeeze(tf.reduce_max(start_terminals, 1), axis=1), 0),
            tf.logical_or(
                tf.greater(tf.squeeze(tf.reduce_max(paths, 1), axis=1), 0),
                tf.greater(tf.squeeze(tf.reduce_max(end_terminals, 1), axis=1), 0))
        )  # (batch, )

        if self.is_evaluating:
            cond = tf.where(any_contexts_is_valid)
        else:  # training
            word_is_valid = tf.greater(entities, 0)  # (batch, )
            cond = tf.where(tf.logical_and(word_is_valid, any_contexts_is_valid))  # (batch, 1)
        valid_mask = tf.to_float(  # (batch, max_contexts, 1)
            tf.logical_or(tf.logical_or(tf.greater(start_terminals, 0),
                                        tf.greater(end_terminals, 0)),
                          tf.greater(paths, 0))
        )

        filtered = \
            tf.gather(entities, cond), \
            tf.squeeze(tf.gather(start_terminals, cond), [1, 3]), \
            tf.squeeze(tf.gather(paths, cond), [1, 3]), \
            tf.squeeze(tf.gather(end_terminals, cond), [1, 3]), \
            tf.squeeze(tf.gather(valid_mask, cond), [1, 3])  # (batch, max_contexts)

        return filtered

    def __enter__(self):
        return self

    def should_stop(self):
        return self.coord.should_stop()

    def __exit__(self, type, value, traceback):
        print('Reader stopping')
        self.coord.request_stop()
        self.coord.join(self.threads)
