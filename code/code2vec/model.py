import tensorflow as tf

import PathContextReader
import numpy as np
import time
from common import common
from loader import Loader


# noinspection PyUnresolvedReferences
class Model:
    topk = 10
    num_batches_to_log = 100

    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()

        self.eval_data_lines = None
        self.eval_queue = None
        self.predict_queue = None

        self.eval_placeholder = None
        self.predict_placeholder = None
        self.predict_top_indices_op, self.predict_top_scores_op, \
        self.predict_original_entities_op, self.attention_weights_op = None, None, None, None

        if config.LOAD_PATH:
            self.load_model(sess=None)
        else:
            loader = Loader(config.DATASET_FOLDER)
            self.methods = loader.load_methods()
            self.nodes = loader.load_nodes()
            self.tokens = loader.load_tokens()
            self.paths = loader.load_paths()
            print('Dataset information loaded.')

    def close_session(self):
        self.sess.close()

    def train(self):
        print('Starting training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        multi_batch_start_time = time.time()
        num_batches_to_evaluate = max(int(
            self.config.NUM_EXAMPLES / self.config.BATCH_SIZE * self.config.SAVE_EVERY_EPOCHS), 1)

        self.queue_thread = PathContextReader.PathContextReader(config=self.config, file_path=self.config.TRAIN_PATH)
        optimizer, train_loss = self.build_training_graph(self.queue_thread.input_tensors())
        self.saver = tf.train.Saver(max_to_keep=self.config.MAX_TO_KEEP)
        self.summary_writer = tf.summary.FileWriter('logs/', graph=self.sess.graph)

        self.initialize_session_variables(self.sess)
        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)
        with self.queue_thread.start(self.sess):
            time.sleep(1)
            print('Started reader...')
            try:
                while True:
                    batch_num += 1
                    _, batch_loss = self.sess.run([optimizer, train_loss])
                    sum_loss += batch_loss
                    if batch_num % self.num_batches_to_log == 0:
                        self.trace(sum_loss, batch_num, multi_batch_start_time)
                        print('Number of waiting examples in queue: %d' % self.sess.run(
                            "shuffle_batch/random_shuffle_queue_Size:0"))
                        sum_loss = 0
                        multi_batch_start_time = time.time()
                    if batch_num % num_batches_to_evaluate == 0:
                        epoch_num = int((batch_num / num_batches_to_evaluate) * self.config.SAVE_EVERY_EPOCHS)
                        save_target = self.config.SAVE_PATH + '_iter' + str(epoch_num)
                        self.save_model(self.sess, save_target)
                        print('Saved after %d epochs in: %s' % (epoch_num, save_target))
                        print('------------------------------------')
                        print('Results of evaluation on test data:')
                        self.evaluate_and_print_results(self.config.TEST_PATH, epoch_num)
                        print('------------------------------------')
                        print('------------------------------------')
                        print('Results of evaluation on train data:')
                        self.evaluate_and_print_results(self.config.TRAIN_PATH, epoch_num)
                        print('------------------------------------')

            except tf.errors.OutOfRangeError:
                print('Done training')

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH)
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sH:%sM:%sS\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / (self.num_batches_to_log * self.config.BATCH_SIZE)
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (batch_num, avg_loss,
                                                                              self.config.BATCH_SIZE * self.num_batches_to_log / (
                                                                                  multi_batch_elapsed if multi_batch_elapsed > 0 else 1)))

    def evaluate(self, file_path):
        eval_start_time = time.time()
        if self.eval_queue is None:
            self.eval_queue = PathContextReader.PathContextReader(config=self.config, file_path=file_path,
                                                                  is_evaluating=True)
            self.eval_placeholder = self.eval_queue.get_input_placeholder()
            self.predict_top_indices_op, self.predict_top_scores_op, \
            self.predict_original_entities_op, self.attention_weights_op = \
                self.build_test_graph(self.eval_queue.get_filtered_batches())
            self.saver = tf.train.Saver()

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
            if self.config.RELEASE:
                release_name = self.config.LOAD_PATH + '.release'
                print('Releasing model, output model: %s' % release_name)
                self.saver.save(self.sess, release_name)
                return None

        if self.eval_data_lines is None:
            print('Loading evaluation data from: ' + file_path)
            self.eval_data_lines = common.load_file_lines(file_path)
            print('Done loading evaluation data')

        with open('log.txt', 'w') as output_file:
            num_correct_predictions = np.zeros(self.topk)
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = \
                np.zeros(self.config.ENTITIES_VOCAB_SIZE, dtype=np.int32), \
                np.zeros(self.config.ENTITIES_VOCAB_SIZE, dtype=np.int32), \
                np.zeros(self.config.ENTITIES_VOCAB_SIZE, dtype=np.int32)
            confuse_matrix = np.zeros((self.config.ENTITIES_VOCAB_SIZE, self.config.ENTITIES_VOCAB_SIZE),
                                      dtype=np.int32)
            rank_matrix = np.zeros((self.config.ENTITIES_VOCAB_SIZE, self.config.ENTITIES_VOCAB_SIZE), dtype=np.float32)
            class_sizes = np.zeros(self.config.ENTITIES_VOCAB_SIZE, dtype=np.int32)

            start_time = time.time()

            for batch in common.split_to_batches(self.eval_data_lines, self.config.TEST_BATCH_SIZE):
                top_indices, top_scores, original_entities = self.sess.run(
                    [self.predict_top_indices_op, self.predict_top_scores_op, self.predict_original_entities_op],
                    feed_dict={self.eval_placeholder: batch})
                # Flatten original names from [[]] to []
                original_entities = [w for l in original_entities for w in l]

                num_correct_predictions = self.update_correct_predictions(num_correct_predictions, output_file,
                                                                          zip(original_entities, top_indices))
                true_positive, false_positive, false_negative = \
                    self.update_per_class_stats(zip(original_entities, top_indices),
                                                true_positive, false_positive, false_negative)

                confuse_matrix = self.compute_confuse_matrix(zip(original_entities, top_indices), confuse_matrix)
                rank_matrix = self.compute_rank_matrix(zip(original_entities, top_indices), rank_matrix, class_sizes)

                total_predictions += len(original_entities)
                total_prediction_batches += 1
                if total_prediction_batches % self.num_batches_to_log == 0:
                    elapsed = time.time() - start_time
                    # start_time = time.time()
                    self.trace_evaluation(output_file, num_correct_predictions, total_predictions, elapsed,
                                          len(self.eval_data_lines))

            print('Done testing, epoch reached')
            output_file.write(str(num_correct_predictions / total_predictions) + '\n')

        for i, size in enumerate(class_sizes):
            rank_matrix[i] /= size
        elapsed = int(time.time() - eval_start_time)
        precision, recall, f1 = self.calculate_results(true_positive, false_positive, false_negative)
        print("Evaluation time: %sH:%sM:%sS" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        del self.eval_data_lines
        self.eval_data_lines = None

        return num_correct_predictions / total_predictions, precision, recall, f1, confuse_matrix, rank_matrix

    def evaluate_and_print_results(self, file_path, epoch_num):
        results, precision, recall, f1, confuse_matrix, rank_matrix = self.evaluate(file_path)
        print('Accuracy after %d epochs: %s' % (epoch_num, results[:5]))
        print('Per class statistics after ' + str(epoch_num) + ' epochs:')
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            print('Class ' + str(i + 1) +
                  ': precision: ' + str(p) +
                  ', recall: ' + str(r) +
                  ', F1: ' + str(f))
        print('Mean precision: ' + str(np.mean(precision)) +
              ', mean recall: ' + str(np.mean(recall)) +
              ', mean F1: ' + str(np.mean(f1)))
        print('Confuse matrix:')
        print(confuse_matrix)
        print('Rank matrix:')
        print(rank_matrix)

    @staticmethod
    def update_per_class_stats(results, true_positive, false_positive, false_negative):
        for original_entity, top_indices in results:
            prediction = top_indices[0]
            if prediction == original_entity:
                true_positive[prediction - 1] += 1
            else:
                false_positive[prediction - 1] += 1
                false_negative[original_entity - 1] += 1
        return true_positive, false_positive, false_negative

    @staticmethod
    def compute_confuse_matrix(results, confuse_matrix):
        for original_entity, top_indices in results:
            prediction = top_indices[0]
            confuse_matrix[original_entity - 1][prediction - 1] += 1
        return confuse_matrix

    @staticmethod
    def compute_rank_matrix(results, rank_matrix, class_sizes):
        for original_entity, top_indices in results:
            class_sizes[original_entity - 1] += 1
            for i, prediction in enumerate(top_indices):
                rank_matrix[original_entity - 1][prediction - 1] += i + 1
        return rank_matrix

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def trace_evaluation(output_file, correct_predictions, total_predictions, elapsed, total_examples):
        state_message = 'Evaluated %d/%d examples...' % (total_predictions, total_examples)
        throughput_message = "Prediction throughput: %d samples/sec" % int(
            total_predictions / (elapsed if elapsed > 0 else 1))
        print(state_message)
        print(throughput_message)

    def update_correct_predictions(self, num_correct_predictions, output_file, results):
        for original_entity, top_indices in results:
            predicted_something = False
            for i, predicted_author in enumerate(top_indices):
                if i == 0:
                    output_file.write(
                        'Original: ' + str(original_entity) + ', predicted 1st: ' + str(predicted_author) + '\n')
                predicted_something = True
                if original_entity == predicted_author:
                    output_file.write('\t\t predicted correctly at rank: ' + str(i + 1) + '\n')
                    for j in range(i, self.topk):
                        num_correct_predictions[j] += 1
                    break
            if not predicted_something:
                output_file.write('No results for predicting: ' + str(original_entity))
        return num_correct_predictions

    def get_vocabs(self, initializer=None, trainable=True):
        tokens_vocab = tf.get_variable('TOKENS_VOCAB',
                                       shape=(self.config.TOKENS_VOCAB_SIZE + 1, self.config.EMBEDDINGS_SIZE),
                                       dtype=tf.float32, initializer=initializer, trainable=trainable)

        paths_vocab = tf.get_variable('PATHS_VOCAB',
                                      shape=(self.config.PATHS_VOCAB_SIZE + 1, self.config.EMBEDDINGS_SIZE),
                                      dtype=tf.float32, initializer=initializer, trainable=trainable)

        return tokens_vocab, paths_vocab

    def build_complex_decision_function(self, weights, trainable=True):
        layer_1 = tf.layers.dense(weights, self.config.EMBEDDINGS_SIZE, activation=tf.nn.tanh,
                                  name='DECISION_1', trainable=trainable)  # (batch, dim)
        layer_2 = tf.layers.dense(layer_1, self.config.EMBEDDINGS_SIZE * 2, activation=tf.nn.tanh,
                                  name='DECISION_2', trainable=trainable)  # (batch, 2 * dim)
        layer_out = tf.layers.dense(layer_2, self.config.ENTITIES_VOCAB_SIZE + 1, activation=None,
                                    name='DECISION_OUT', trainable=trainable)  # (batch, entities)
        return layer_out

    def build_simple_decision_function(self, weights, trainable=True):
        # (batch, entities)
        layer_out = tf.layers.dense(weights, self.config.ENTITIES_VOCAB_SIZE + 1, activation=None,
                                    name='DECISION_OUT', trainable=trainable)
        return layer_out

    @staticmethod
    def create_dicts(starts, paths, ends, mask):
        return {'starts': starts,
                'paths': paths,
                'ends': ends,
                'mask': mask}

    def build_training_graph(self, input_tensors):
        entities_input, \
        added_starts, added_paths, added_ends, added_mask, \
        deleted_starts, deleted_paths, deleted_ends, deleted_mask \
            = input_tensors  # (batch, 1), (batch, pack, max_contexts)
        added = self.create_dicts(added_starts, added_paths, added_ends, added_mask)
        deleted = self.create_dicts(deleted_starts, deleted_paths, deleted_ends, deleted_mask)

        with tf.variable_scope('model'):
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True)
            tokens_vocab, paths_vocab = self.get_vocabs(initializer=initializer)

            # (batch, pack, max_contexts * 2, dim)
            contexts_embed, valid_mask = \
                self.build_contexts(tokens_vocab, paths_vocab, added, deleted)

            # (batch, pack, dim)
            weighted_average_contexts, _ = \
                self.calculate_weighted_contexts(contexts_embed, valid_mask)

            # (batch, dim)
            weighted_average_methods, _ = \
                self.calculate_weighted_methods(weighted_average_contexts)

            # (batch, entities)
            logits = self.build_simple_decision_function(weighted_average_contexts)

            batch_size = tf.to_float(tf.shape(entities_input)[0])
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(entities_input, [-1]),
                logits=logits)) / batch_size

            optimizer = tf.train.AdamOptimizer().minimize(loss)

        return optimizer, loss

    def build_test_graph(self, input_tensors, normalize_scores=False):
        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            tokens_vocab, paths_vocab = self.get_vocabs(trainable=False)
            entities_input, \
            added_starts, added_paths, added_ends, added_mask, \
            deleted_starts, deleted_paths, deleted_ends, deleted_mask = input_tensors
            added = self.create_dicts(added_starts, added_paths, added_ends, added_mask)
            deleted = self.create_dicts(deleted_starts, deleted_paths, deleted_ends, deleted_mask)
            
            # (batch, pack, max_contexts * 2, dim)
            contexts_embed, valid_mask = \
                self.build_contexts(tokens_vocab, paths_vocab, added, deleted, trainable=False)
            
            # (batch, pack, dim)
            weighted_average_contexts, _ = \
                self.calculate_weighted_contexts(contexts_embed, valid_mask, trainable=False)

            # (batch, dim)
            weighted_average_methods, _ = \
                self.calculate_weighted_methods(weighted_average_contexts, trainable=False)

            # (batch, entities)
            cos = self.build_simple_decision_function(weighted_average_methods, trainable=False)

        topk_candidates = tf.nn.top_k(cos, k=tf.minimum(self.topk, self.config.ENTITIES_VOCAB_SIZE))
        top_indices = tf.to_int64(topk_candidates.indices)
        original_entities = entities_input
        top_scores = topk_candidates.values
        if normalize_scores:
            top_scores = tf.nn.softmax(top_scores)

        return top_indices, top_scores, original_entities, attention_weights

    def build_contexts(self, tokens_vocab, paths_vocab, removed, added, trainable=True):
        keep_prob1 = 0.75

        # (batch, pack, max_contexts * 2, 1)
        starts = tf.concat([removed['starts'], added['starts']], axis=2)
        paths = tf.concat([removed['paths'], added['paths']], axis=2)
        ends = tf.concat([removed['ends'], added['ends']], axis=2)
        # (batch, pack, max_contexts * 2)
        valid_mask = tf.concat([removed['mask'], added['mask']], axis=2)

        # (batch, pack, max_contexts * 2, dim)
        start_token_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=starts)
        path_embed = tf.nn.embedding_lookup(params=paths_vocab, ids=paths)
        end_token_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=ends)

        # (batch, pack, max_contexts * 2, dim * 3)
        context_embed = tf.concat([start_token_embed, path_embed, end_token_embed], axis=-1)

        if trainable:
            context_embed = tf.nn.dropout(context_embed, keep_prob1)

        transform_param = tf.get_variable('TRANSFORM',
                                          shape=(self.config.EMBEDDINGS_SIZE * 3, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=trainable)

        # (batch, pack, max_contexts * 2, dim)
        transformed_embed = tf.tanh(tf.matmul(context_embed, transform_param))
        return transformed_embed, valid_mask

    def calculate_weighted_contexts(self, context_embed, valid_mask, trainable=True):
        # input -> (batch, pack, max_contexts * 2, dim)

        # (batch, pack, max_contexts * 2, 1)
        contexts_weights = tf.layers.dense(context_embed, 1, activation=None,
                                          name='ATTENTION_CONTEXTS', trainable=trainable)

        # (batch, pack, max_contexts * 2)
        mask = tf.log(valid_mask)

        # (batch, pack, max_contexts * 2, 1)
        mask = tf.expand_dims(mask, axis=3)
        contexts_weights += mask
        attention_weights = tf.nn.softmax(batched_contexts_weights, axis=2)

        # (batch, pack, dim)
        weighted_average_contexts = tf.reduce_sum(tf.multiply(context_embed, attention_weights), axis=2)

        return weighted_average_contexts, attention_weights

    def calculate_weighted_methods(self, method_embed, trainable=True):
        # input -> (batch, pack, dim)

        # (batch, pack, 1)
        methods_weights = tf.layers.dense(method_embed, 1, activation=None,
                                          name='ATTENTION_METHODS', trainable=trainable)

        # (batch, pack, 1)
        attention_weights = tf.nn.softmax(methods_weights, axis=1)

        # (batch, dim)
        weighted_average_contexts = tf.reduce_sum(tf.multiply(method_embed, attention_weights), axis=1)

        return weighted_average_contexts, attention_weights

    def predict(self, predict_data_lines):
        if self.predict_queue is None:
            self.predict_queue = PathContextReader.PathContextReader(config=self.config,
                                                                     file_path=self.config.TEST_PATH,
                                                                     is_evaluating=True)
            self.predict_placeholder = self.predict_queue.get_input_placeholder()
            self.predict_top_indices_op, self.predict_top_scores_op, \
            self.predict_original_entities_op, self.attention_weights_op = \
                self.build_test_graph(self.predict_queue.get_filtered_batches(), normalize_scores=True)

            self.initialize_session_variables(self.sess)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)

        results = []
        for batch in common.split_to_batches(predict_data_lines, 1):
            top_words, top_scores, original_names, attention_weights, source_strings, path_strings, target_strings = self.sess.run(
                [self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op,
                 self.attention_weights_op, self.predict_source_string, self.predict_path_string,
                 self.predict_path_target_string],
                feed_dict={self.predict_placeholder: batch})
            top_words, original_names = common.binary_to_string_matrix(top_words), common.binary_to_string_matrix(
                original_names)
            # Flatten original names from [[]] to []
            attention_per_path = self.get_attention_per_path(source_strings, path_strings, target_strings,
                                                             attention_weights)
            original_names = [w for l in original_names for w in l]
            results.append((original_names[0], top_words[0], top_scores[0], attention_per_path))
        return results

    def get_attention_per_path(self, source_strings, path_strings, target_strings, attention_weights):
        attention_weights = np.squeeze(attention_weights)  # (max_contexts, )
        attention_per_context = {}
        for source, path, target, weight in zip(source_strings, path_strings, target_strings, attention_weights):
            string_triplet = (
                common.binary_to_string(source), common.binary_to_string(path), common.binary_to_string(target))
            attention_per_context[string_triplet] = weight
        return attention_per_context

    def save_model(self, sess, path):
        self.saver.save(sess, path)

    def load_model(self, sess):
        if not sess is None:
            print('Loading model weights from: ' + self.config.LOAD_PATH)
            self.saver.restore(sess, self.config.LOAD_PATH)
            print('Done')

        loader = Loader(self.config.DATASET_FOLDER)
        self.methods = loader.load_methods()
        self.nodes = loader.load_nodes()
        self.tokens = loader.load_tokens()
        self.paths = loader.load_paths()
        print('Dataset information loaded.')

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))

    def get_should_reuse_variables(self):
        if self.config.TRAIN_PATH:
            return True
        else:
            return None
