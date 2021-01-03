from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_distribution_strategy(distribution_strategy="mirrored",
                              num_gpus=0,
                              all_reduce_alg=None,
                              num_packs=1,
                              tpu_address=None):
    if num_gpus < 0:
        raise ValueError("`num_gpus` can not be negative")

    distribution_strategy = distribution_strategy.lower()
    if distribution_strategy == "off":
        if num_gpus > 1:
            raise ValueError("When {} GPUs are specified, distribution_strategy flag cannot be set to `off`."
                             .format(num_gpus))
        return None

    if distribution_strategy == "tpu":
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
        if tpu_address not in ("", "local"):
            tf.config.experimental_connect_to_cluster(cluster_spec_or_resolver=cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver=cluster_resolver)
        return tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver=cluster_resolver)

    if distribution_strategy == "multi_worker_mirrored":
        communications_dict = {
            None: tf.distribute.experimental.CollectiveCommunication.AUTO,
            "ring": tf.distribute.experimental.CollectiveCommunication.RING,
            "nccl": tf.distribute.experimental.CollectiveCommunication.NCCL
        }
        return tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=communications_dict[all_reduce_alg])

    if distribution_strategy == "one_device":
        if num_gpus == 0:
            return tf.distribute.OneDeviceStrategy("device:CPU:0")
        if num_gpus > 1:
            raise ValueError("`OneDeviceStrategy` can not be used for more than one device.")
        return tf.distribute.OneDeviceStrategy("device:GPU:0")

    if distribution_strategy == "mirrored":
        if num_gpus == 0:
            devices = ["device:CPU:0"]
        else:
            devices = ["device:GPU:%d" % i for i in range(num_gpus)]
        if all_reduce_alg is None:
            cross_device_ops = None
        else:
            mirrored_all_reduce_options = {
                "nccl": tf.distribute.NcclAllReduce,
                "hierarchical_copy": tf.distribute.HierarchicalCopyAllReduce,
                # "reduction_to_one_device": tf.distribute.ReductionToOneDevice
            }
            cross_device_ops = mirrored_all_reduce_options[all_reduce_alg](num_packs=num_packs)
        return tf.distribute.MirroredStrategy(devices=devices, cross_device_ops=cross_device_ops)

    if distribution_strategy == "parameter_server":
        return tf.distribute.experimental.ParameterServerStrategy()

    if distribution_strategy == "central_storage":
        return tf.distribute.experimental.CentralStorageStrategy()

    raise ValueError("Unrecognized Distribution Strategy: %r" % distribution_strategy)


def get_strategy_scope(strategy):
    if strategy:
        strategy_scope = strategy.scope()
    else:
        strategy_scope = DummyContextManager()
    return strategy_scope


class DummyContextManager(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
