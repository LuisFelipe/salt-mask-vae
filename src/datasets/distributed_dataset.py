# coding=utf-8
"""
module dataset.py
__________________________________
Base dataset wrapper definition.
"""
import tensorflow as tf
from flow.config import Config
from flow.callbacks import on_epoch_begin, before_session_initialization, on_validate_begin
from blinker import signal


class DistributedDataset(object):

    def __init__(self, partitions: dict, inputs_config, strategy: tf.distribute.MirroredStrategy, config: Config=None, *args, **kwargs):
        # configurations
        self.partitions = partitions
        self.current_partition = "train"
        self.config = config
        self._inputs_config = inputs_config
        self.strategy = strategy
        # build dataset
        self._dataset = self.build_dataset()
        
        on_epoch_begin.connect(
            lambda sender: self.initialize_iterator(sender, "train"),
            weak=False
        )
        on_validate_begin.connect(
            lambda sender: self.initialize_iterator(sender, "valid"),
            weak=False
        )

    def __iter__(self):
        if self.current_partition == "train":
            return iter(self.partitions["train"])
        elif self.current_partition == "valid":
            return iter(self.partitions["valid"])
        elif self.current_partition == "test":
            return iter(self.partitions["test"])

    def __next__(self):
        if self.current_partition == "train":
            return next(self.partitions["train"])
        elif self.current_partition == "valid":
            return next(self.partitions["valid"])
        elif self.current_partition == "test":
            return next(self.partitions["test"])

    def __len__(self):
        if self.current_partition == "train":
            return len(self.partitions["train"])
        elif self.current_partition == "valid":
            return len(self.partitions["valid"])
        elif self.current_partition == "test":
            return len(self.partitions["test"])

    def initialize_iterator(self, sender, partition):
        """
        Initializes the current dataset iterator by runing the iterator initializer on the current session.
        :param sender: the function caller object
        """
        self.current_partition = partition
        if tf.distribute.get_replica_context() is not None:
            with self.strategy.scope():
                current_session = tf.get_default_session()
                current_session.run(self._iterator_initializer)
        else:
            current_session = tf.get_default_session()
            current_session.run(self._iterator_initializer)

    def restart(self):
        """
        Restart iterations from the first sequence element.
        **A hook to 'self.initialize_iterator'.**
        """
        self.initialize_iterator(None)

    def build_dataset(self):
        def _build():
            batch_size = int(self.config.get("FLOW.BATCH_SIZE", 1))
            prefetch_buffer = 100
            dataset = tf.data.Dataset.from_generator(
                generator=lambda: iter(self),
                output_types=self._inputs_config["output_types"],
                output_shapes=self._inputs_config["output_shapes"]
            )
            dataset = dataset.batch(batch_size * self.strategy.num_replicas_in_sync)
            dataset = dataset.prefetch(buffer_size=prefetch_buffer)
            self._dataset = self.strategy.experimental_distribute_dataset(dataset)


            self._iterator = self._dataset.make_initializable_iterator()
            self._iterator_initializer = self._iterator.initialize()
            self.next_elements = next(self._iterator)

            return self._dataset
        if tf.distribute.get_replica_context() is not None:
            with self.strategy.scope():
                return _build()
        else:
            return _build()
