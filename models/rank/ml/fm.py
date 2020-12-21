import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import CategoricalColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List


class FMModel(tf.keras.Model):
    def __init__(self, latent_dim: int, columns: List[CategoricalColumn], name=None, **kwargs):
        super(FMModel, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.columns = sorted(columns, key=lambda col: col.name)
        self.state_manager = _StateManagerImplV2(self, True)
        self.bias = None

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_columns
        config = {
            "latent_dim": self.latent_dim,
            "columns": serialize_feature_columns(self.columns)
        }
        base_config = super(FMModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        for column in self.columns:
            self.state_manager.create_variable(
                feature_column=column,
                name="weights",
                shape=(column.num_buckets, 1),
                dtype=self.dtype,
                trainable=True)
            self.state_manager.create_variable(
                feature_column=column,
                name="embeddings",
                shape=(column.num_buckets, self.latent_dim),
                dtype=self.dtype,
                trainable=True)
        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        linear_tensors = []
        embedding_tensors = []
        for column in self.columns:
            sparse_tensors = column.get_sparse_tensors(transformation_cache, None)
            linear_tensor = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=self.state_manager.get_variable(column, "weights"),
                sparse_ids=sparse_tensors.id_tensor,
                sparse_weights=sparse_tensors.weight_tensor,
                combiner="sum")
            linear_tensors.append(linear_tensor)
            embedding_tensor = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=self.state_manager.get_variable(column, "embeddings"),
                sparse_ids=sparse_tensors.id_tensor,
                sparse_weights=sparse_tensors.weight_tensor,
                combiner="mean")
            embedding_tensors.append(embedding_tensor)
        linear_logits_no_bias = tf.math.reduce_sum(tf.math.add_n(linear_tensors), axis=1, keepdims=True)
        linear_logits = tf.nn.bias_add(linear_logits_no_bias, self.bias)
        embedding_tensor = tf.stack(embedding_tensors, axis=1)
        embedding_sum_square = tf.math.square(tf.math.reduce_sum(embedding_tensor, axis=1))
        embedding_square_sum = tf.math.reduce_sum(tf.math.square(embedding_tensor), axis=1)
        embedding_logits = tf.math.reduce_sum(embedding_sum_square - embedding_square_sum, axis=1) * 0.5
        logits = linear_logits + embedding_logits
        predict = tf.keras.activations.sigmoid(logits)
        return predict


def main(_):
    import os
    titanic_train_data_path = os.path.abspath(__file__).replace("models/rank/ml/fm.py", "data/titanic/train.csv")

    train_data = tf.data.experimental.make_csv_dataset(
        file_pattern=titanic_train_data_path,
        batch_size=10,
        column_names=["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare",
                      "Cabin", "Embarked"],
        column_defaults=[0, 0, "", "", 0.0, 0, 0, "", 0.0, "", ""],
        label_name="Survived",
        select_columns=["Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin",
                        "Embarked"],
        field_delim=",",
        use_quote_delim=True,
        na_value="",
        header=True,
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=1000,
        num_rows_for_inference=0,
        ignore_errors=False)

    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(key="Pclass", vocabulary_list=[1, 2, 3]),
        tf.feature_column.categorical_column_with_vocabulary_list(key="Sex", vocabulary_list=["female", "male"]),
        tf.feature_column.bucketized_column(source_column=tf.feature_column.numeric_column(key="Age"),
                                            boundaries=[1, 6, 14, 18, 30, 40, 50, 60, 70]),
    ]

    model = FMModel(latent_dim=10, columns=feature_columns, l2_factor=0.5)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  loss=tf.keras.losses.binary_crossentropy)
    model.fit(x=train_data, epochs=10)
    tf.keras.utils.plot_model(model, to_file="lr.png")


if __name__ == "__main__":
    app.run(main)
