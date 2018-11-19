import tensorflow as tf
import common

no_such_word = 0
no_such_composite = '{} {} {}'.format(no_such_word, no_such_word, no_such_word)


class PathContextReader:
    class_word_table = None
    class_target_word_table = None
    class_path_table = None

    def __init__(self, config, file_path, is_evaluating=False):
        # self.file_path = config.TEST_PATH if is_evaluating else config.TRAIN_PATH
        self.file_path = file_path
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
        record_defaults = [[no_such_composite]] * (self.max_contexts * 3 + 1)
        row_parts = tf.decode_csv(row, record_defaults=record_defaults, field_delim=',')
        entities = tf.string_to_number(row_parts[0], out_type=tf.int32)  # (batch, )
        all_contexts = tf.stack(row_parts[1:(3 * self.max_contexts + 1)], axis=1)  # (batch, 3 * max_contexts)

        flat_contexts = tf.reshape(all_contexts, [-1])  # (batch * 3 * max_contexts, )
        split_contexts = tf.string_split(flat_contexts, delimiter=' ')
        dense_split_contexts = tf.reshape(tf.sparse_tensor_to_dense(split_contexts,
                                                                    default_value=str(no_such_word)),
                                          shape=[-1, 3 * self.max_contexts, 3])  # (batch, 3 * max_contexts, 3)

        dense_split_contexts = tf.string_to_number(dense_split_contexts, out_type=tf.int32)
        starts = tf.slice(dense_split_contexts, [0, 0, 0], [-1, 3 * self.max_contexts, 1])
        paths = tf.slice(dense_split_contexts, [0, 0, 1], [-1, 3 * self.max_contexts, 1])
        ends = tf.slice(dense_split_contexts, [0, 0, 2], [-1, 3 * self.max_contexts, 1])

        added_starts, deleted_starts, _ = tf.split(starts, 3, axis=1)
        added_paths, deleted_paths, _ = tf.split(paths, 3, axis=1)
        added_ends, deleted_ends, _ = tf.split(ends, 3, axis=1)

        return entities, added_starts, added_paths, added_ends, deleted_starts, deleted_paths, deleted_ends

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
        entities, added_starts, added_paths, added_ends, deleted_starts, deleted_paths, deleted_ends = self.read_file()

        def any_valid(t1, t2, t3):
            return tf.logical_or(
                tf.greater(tf.squeeze(tf.reduce_max(t1, 1), axis=1), 0),
                tf.logical_or(
                    tf.greater(tf.squeeze(tf.reduce_max(t2, 1), axis=1), 0),
                    tf.greater(tf.squeeze(tf.reduce_max(t3, 1), axis=1), 0))
            )

        any_contexts_is_valid = tf.logical_or(
            any_valid(added_starts, added_paths, added_ends),
            any_valid(deleted_starts, deleted_paths, deleted_ends)
        )  # (batch, )

        if self.is_evaluating:
            cond = tf.where(any_contexts_is_valid)
        else:  # training
            entity_is_valid = tf.greater(entities, 0)  # (batch, )
            cond = tf.where(tf.logical_and(entity_is_valid, any_contexts_is_valid))  # (batch, 1)

        def valid_mask(t1, t2, t3):
            return tf.to_float(
                tf.logical_or(tf.logical_or(tf.greater(t1, 0),
                                            tf.greater(t2, 0)),
                              tf.greater(t3, 0))
            )

        added_valid_mask = valid_mask(added_starts, added_paths, added_ends)  # (batch, max_contexts, 1)
        deleted_valid_mask = valid_mask(deleted_starts, deleted_paths, deleted_ends)  # (batch, max_contexts, 1)

        filtered = \
            tf.gather(entities, cond), \
            tf.squeeze(tf.gather(added_starts, cond), [1, 3]), \
            tf.squeeze(tf.gather(added_paths, cond), [1, 3]), \
            tf.squeeze(tf.gather(added_ends, cond), [1, 3]), \
            tf.squeeze(tf.gather(added_valid_mask, cond), [1, 3]), \
            tf.squeeze(tf.gather(deleted_starts, cond), [1, 3]), \
            tf.squeeze(tf.gather(deleted_paths, cond), [1, 3]), \
            tf.squeeze(tf.gather(deleted_ends, cond), [1, 3]), \
            tf.squeeze(tf.gather(deleted_valid_mask, cond), [1, 3])

        return filtered

    def __enter__(self):
        return self

    def should_stop(self):
        return self.coord.should_stop()

    def __exit__(self, type, value, traceback):
        print('Reader stopping')
        self.coord.request_stop()
        self.coord.join(self.threads)
