import tensorflow as tf

from PackDataset import PackDataset
from ContextsLoader import ContextsLoader
import numpy as np
import time
from loader import Loader


# noinspection PyUnresolvedReferences
class Model:
    topk = 16
    num_batches_to_log = 100

    def __init__(self, config):
        np.set_printoptions(precision=3, suppress=True)
        self.config = config
        self.sess = tf.Session()

        self.eval_data_lines = None
        self.eval_queue = None
        self.predict_queue = None

        self.predict_placeholder = None
        self.predict_top_indices_op, self.predict_top_scores_op, \
        self.predict_original_entities_op, self.attention_weights_op = None, None, None, None

        if config.LOAD_PATH:
            self.load_model(sess=None)
        # else:
        #     loader = Loader(config.DATASET_FOLDER)
        #     self.methods = loader.load_methods()
        #     self.nodes = loader.load_nodes()
        #     self.tokens = loader.load_tokens()
        #     self.paths = loader.load_paths()
        #     print('Dataset information loaded.')

    def close_session(self):
        self.sess.close()

    def init_loaders(self):
        self.pack_dataset = PackDataset(self.config, self.config.TRAIN_PATH, self.config.TEST_PATH)
        self.TRAIN_EXAMPLES = self.pack_dataset.train_examples
        self.TEST_EXAMPLES = self.pack_dataset.test_examples
        self.config.ENTITIES_VOCAB_SIZE = self.pack_dataset.entities_cnt
        self.topk = self.config.ENTITIES_VOCAB_SIZE + 1
        print('Created pack dataset')
        self.contexts_loader = ContextsLoader(self.config, self.config.CHANGES_PATH)
        print('Created contexts loader')

    def train(self):
        print('Starting training')
        start_time = time.time()

        self.init_loaders()

        optimizer, train_loss = self.build_training_graph()
        print('Built training graph')

        self.predict_top_indices_op, self.predict_top_scores_op, \
        self.predict_original_entities_op, self.attention_weights_op = self.build_test_graph()
        print('Built test graph')

        self.saver = tf.train.Saver(max_to_keep=self.config.MAX_TO_KEEP)
        self.summary_writer = tf.summary.FileWriter('logs/', graph=self.sess.graph)

        self.initialize_session_variables(self.sess)
        print('Initalized variables')

        if self.config.LOAD_PATH:
            self.load_model(self.sess)

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print("Epoch #{}".format(epoch))

            self.pack_dataset.init_epoch()
            batch_num = 0
            sum_loss = 0
            multi_batch_start_time = time.time()

            for batched_packs, batched_entities in self.pack_dataset.train_generator:
                batch_num += 1
                packs_before, packs_after = self.contexts_loader.get(batched_packs)
                _, batch_loss = self.sess.run([optimizer, train_loss], feed_dict={
                    self.pack_dataset.packs_before_placeholder: packs_before,
                    self.pack_dataset.packs_after_placeholder: packs_after,
                    self.pack_dataset.entities_placeholder: batched_entities
                })
                sum_loss += batch_loss
                if batch_num % self.num_batches_to_log == 0:
                    self.trace(sum_loss, batch_num, multi_batch_start_time)
                    sum_loss = 0
                    multi_batch_start_time = time.time()

            if epoch % self.config.SAVE_EVERY_EPOCHS == 0:
                save_target = self.config.SAVE_PATH + '_iter' + str(epoch)
                self.save_model(self.sess, save_target)
                print('Saved after %d epochs in: %s' % (epoch, save_target))
                print('------------------------------------')
                print('Results of evaluation on test data:')
                self.evaluate_and_print_results(epoch)
                print('------------------------------------')

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

    def evaluate_and_print_results(self, epoch_num):
        results, precision, recall, f1, confuse_matrix, rank_matrix = self.evaluate()
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

    def evaluate(self):
        eval_start_time = time.time()
        with open('log.txt', 'w') as output_file:
            num_correct_predictions = np.zeros(self.topk)
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = \
                np.zeros(self.config.ENTITIES_VOCAB_SIZE, dtype=np.int32), \
                np.zeros(self.config.ENTITIES_VOCAB_SIZE, dtype=np.int32), \
                np.zeros(self.config.ENTITIES_VOCAB_SIZE, dtype=np.int32)

            confuse_matrix = np.zeros((self.config.ENTITIES_VOCAB_SIZE, self.config.ENTITIES_VOCAB_SIZE + 1),
                                      dtype=np.int32)
            rank_matrix = np.zeros((self.config.ENTITIES_VOCAB_SIZE, self.config.ENTITIES_VOCAB_SIZE + 1),
                                   dtype=np.float32)
            class_sizes = np.zeros(self.config.ENTITIES_VOCAB_SIZE, dtype=np.int32)

            start_time = time.time()

            for batched_packs, batched_entities in self.pack_dataset.test_generator:
                packs_before, packs_after = self.contexts_loader.get(batched_packs)
                top_indices, top_scores, original_entities = self.sess.run(
                    [self.predict_top_indices_op, self.predict_top_scores_op, self.predict_original_entities_op],
                    feed_dict={
                        self.pack_dataset.packs_before_placeholder: packs_before,
                        self.pack_dataset.packs_after_placeholder: packs_after,
                        self.pack_dataset.entities_placeholder: batched_entities
                    })

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
                    self.trace_evaluation(total_predictions, elapsed, self.TEST_EXAMPLES)

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
            confuse_matrix[original_entity - 1][prediction] += 1
        return confuse_matrix

    @staticmethod
    def compute_rank_matrix(results, rank_matrix, class_sizes):
        for original_entity, top_indices in results:
            class_sizes[original_entity - 1] += 1
            for i, prediction in enumerate(top_indices):
                rank_matrix[original_entity - 1][prediction] += i + 1
        return rank_matrix

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def trace_evaluation(total_predictions, elapsed, total_examples):
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

    def build_training_graph(self):
        # packs -> (batch, )
        # entities -> (batch, pack, max_contexts, dim)
        before_contexts, after_contexts, entities = \
            self.pack_dataset.packs_before_placeholder, \
            self.pack_dataset.packs_after_placeholder, \
            self.pack_dataset.entities_placeholder

        with tf.variable_scope('model'):
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True)
            tokens_vocab, paths_vocab = self.get_vocabs(initializer=initializer)

            # (batch, pack, max_contexts * 2, dim)
            # contexts_embed, valid_mask = \
            #     self.build_contexts(tokens_vocab, paths_vocab, before_contexts, after_contexts)
            contexts_embed = \
                self.build_contexts(tokens_vocab, paths_vocab, before_contexts, after_contexts)

            # (batch, pack, dim)
            weighted_average_contexts, _ = \
                self.calculate_weighted_contexts(contexts_embed, None)

            # (batch, dim)
            weighted_average_methods, _ = \
                self.calculate_weighted_methods(weighted_average_contexts)

            # (batch, entities)
            logits = self.build_simple_decision_function(weighted_average_methods)

            batch_size = tf.to_float(tf.shape(entities)[0])
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(entities, [-1]),
                logits=logits)) / batch_size

            optimizer = tf.train.AdamOptimizer().minimize(loss)

        return optimizer, loss

    def build_test_graph(self, normalize_scores=False):
        # packs -> (batch, )
        # entities -> (batch, pack, max_contexts, dim)
        before_contexts, after_contexts, entities = \
            self.pack_dataset.packs_before_placeholder, \
            self.pack_dataset.packs_after_placeholder, \
            self.pack_dataset.entities_placeholder

        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            tokens_vocab, paths_vocab = self.get_vocabs(trainable=False)

            # (batch, pack, max_contexts * 2, dim)
            # contexts_embed, valid_mask = \
            #     self.build_contexts(tokens_vocab, paths_vocab, added, deleted, trainable=False)
            contexts_embed = \
                self.build_contexts(tokens_vocab, paths_vocab, before_contexts, after_contexts, trainable=False)

            # (batch, pack, dim)
            weighted_average_contexts, _ = \
                self.calculate_weighted_contexts(contexts_embed, None, trainable=False)
            # (batch, dim)
            weighted_average_methods, attention_weights = \
                self.calculate_weighted_methods(weighted_average_contexts, trainable=False)

            # (batch, entities)
            cos = self.build_simple_decision_function(weighted_average_methods, trainable=False)

        topk_candidates = tf.nn.top_k(cos, k=tf.minimum(self.topk, self.config.ENTITIES_VOCAB_SIZE))
        top_indices = tf.to_int64(topk_candidates.indices)
        original_entities = entities
        top_scores = topk_candidates.values
        if normalize_scores:
            top_scores = tf.nn.softmax(top_scores)

        return top_indices, top_scores, original_entities, attention_weights

    def get_slice(self, contexts, dim):
        return tf.slice(contexts, [0, 0, 0, dim], [-1, -1, self.config.MAX_CONTEXTS, 1])

    def build_contexts(self, tokens_vocab, paths_vocab, before_contexts, after_contexts, trainable=True):
        # input -> (batch, pack, max_contexts, dim)

        keep_prob1 = 0.75

        # (batch, pack, max_contexts * 2, 1)
        starts = tf.concat([self.get_slice(before_contexts, 0), self.get_slice(after_contexts, 0)], axis=2)
        paths = tf.concat([self.get_slice(before_contexts, 1), self.get_slice(after_contexts, 1)], axis=2)
        ends = tf.concat([self.get_slice(before_contexts, 2), self.get_slice(after_contexts, 2)], axis=2)
        starts = tf.squeeze(starts, axis=-1)
        paths = tf.squeeze(paths, axis=-1)
        ends = tf.squeeze(ends, axis=-1)
        # (batch, pack, max_contexts * 2)
        # valid_mask = tf.concat([removed['mask'], added['mask']], axis=2)

        # (batch, pack, max_contexts * 2, dim)
        start_token_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=starts)
        path_embed = tf.nn.embedding_lookup(params=paths_vocab, ids=paths)
        end_token_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=ends)

        # (batch, pack, max_contexts * 2, dim * 3)
        context_embed = tf.concat([start_token_embed, path_embed, end_token_embed], axis=-1)

        if trainable:
            context_embed = tf.nn.dropout(context_embed, keep_prob1)

        # (batch, pack, max_contexts * 2, dim)
        transformed_embed = tf.layers.dense(context_embed, self.config.EMBEDDINGS_SIZE, activation=None,
                                            name='TRANSFORM', trainable=trainable)

        return transformed_embed  # , valid_mask

    def calculate_weighted_contexts(self, context_embed, valid_mask, trainable=True):
        # input -> (batch, pack, max_contexts * 2, dim)

        # (batch, pack, max_contexts * 2, 1)
        contexts_weights = tf.layers.dense(context_embed, 1, activation=None,
                                           name='ATTENTION_CONTEXTS', trainable=trainable)

        # (batch, pack, max_contexts * 2)
        # mask = tf.log(valid_mask)

        # (batch, pack, max_contexts * 2, 1)
        # mask = tf.expand_dims(mask, axis=3)
        # contexts_weights += mask
        attention_weights = tf.nn.softmax(contexts_weights, axis=2)

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
