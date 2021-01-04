import tensorflow as tf


def build_parsing_serving_input_receiver_fn(feature_spec, default_batch_size=None):
    def serving_input_receiver_fn():
        serialized_tf_example = tf.compat.v1.placeholder(
            dtype=tf.string,
            shape=[default_batch_size],
            name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.compat.v1.io.parse_example(serialized_tf_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    return serving_input_receiver_fn


# 模型训练结束导出最后一份模型
def model_final_exporter(model_name, feature_schema):
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_schema)
    return tf.estimator.FinalExporter(name=model_name,
                                      serving_input_receiver_fn=serving_input_receiver_fn)


# 评估效果超过之前所有已存在的模型效果，就导出模型
def model_best_exporter(model_name, feature_schema, exports_to_keep=1, metric_key="loss", big_better=True):
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_schema)

    def compare(best_eval_result, current_eval_result):
        if not best_eval_result or metric_key not in best_eval_result:
            raise ValueError(
                'best_eval_result cannot be empty or no loss is found in it.')

        if not current_eval_result or metric_key not in current_eval_result:
            raise ValueError(
                'current_eval_result cannot be empty or no loss is found in it.')

        if big_better:
            return best_eval_result[metric_key] < current_eval_result[metric_key]
        else:
            return best_eval_result[metric_key] > current_eval_result[metric_key]

    return tf.estimator.BestExporter(name=model_name,
                                     serving_input_receiver_fn=serving_input_receiver_fn,
                                     compare_fn=compare,
                                     exports_to_keep=exports_to_keep)


# 每次评估都导出模型，默认最多保存3份
def model_latest_exporter(model_name, feature_schema, exports_to_keep=3):
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_schema)
    return tf.estimator.LatestExporter(name=model_name,
                                       exports_to_keep=exports_to_keep,
                                       serving_input_receiver_fn=serving_input_receiver_fn)
