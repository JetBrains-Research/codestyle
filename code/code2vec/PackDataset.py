import tensorflow as tf
import numpy as np


class PackDataset:

    def __init__(self, config, train_files, test_files):
        self.config = config
        self.train_entities, self.train_packs = self.read_files(train_files, shuffle=True)
        self.test_entities, self.test_packs = self.read_files(test_files)
        self.train_entities_placeholder, self.train_packs_placeholder = \
            self.create_placeholders(self.train_entities, self.train_packs)
        self.test_entities_placeholder, self.test_packs_placeholder = \
            self.create_placeholders(self.test_entities, self.test_packs)
        self.train_dataset, self.train_iterator = \
            self.create_dataset(self.train_packs_placeholder, self.train_entities_placeholder)
        self.test_dataset, self.test_iterator = \
            self.create_dataset(self.test_packs_placeholder, self.test_entities_placeholder)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_train = self.train_iterator.get_next()
        self.next_test = self.test_iterator.get_next()

    def read_files(self, files, shuffle=False):
        entities = []
        packs = []
        for filename in files:
            with open(filename, 'r') as fin:
                for line in fin:
                    items = list(map(int, line.split(',')))
                    entities.append(items[0])
                    packs.append(items[1:])

        entities = np.array(entities, dtype=np.int32)
        packs = np.array(packs, dtype=np.int32)
        if shuffle:
            perm = np.random.permutation(len(entities))
            # entities = entities[perm]
            # packs = packs[perm]
        return entities, packs

    def create_placeholders(self, entities, packs):
        return tf.placeholder(entities.dtype, entities.shape), tf.placeholder(packs.dtype, packs.shape)

    def create_dataset(self, packs_placeholder, entities_placeholder):
        dataset = tf.data.Dataset.from_tensor_slices((packs_placeholder, entities_placeholder))
        dataset = dataset.batch(self.config.BATCH_SIZE)
        dataset = dataset.repeat(self.config.NUM_EPOCHS)
        iterator = dataset.make_initializable_iterator()
        return dataset, iterator

    def init_iterators(self, sess):
        # initialise iterators
        sess.run(self.train_iterator.initializer, feed_dict={
            self.train_packs_placeholder: self.train_packs,
            self.train_entities_placeholder: self.train_entities
        })
        sess.run(self.test_iterator.initializer, feed_dict={
            self.test_packs_placeholder: self.test_packs,
            self.test_entities_placeholder: self.test_entities
        })
    #
    # def next_train(self, sess):
    #     return sess.run(self.next_elements, feed_dict={self.handle: self.train_handle})
    #
    # def next_test(self, sess):
    #     return sess.run(self.next_elements, feed_dict={self.handle: self.test_handle})
