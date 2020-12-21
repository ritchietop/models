import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import DenseColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List
from absl import app


class DNNModel(tf.keras.Model):
    def __init__(self,
                 output_dim: int,
                 feature_columns: List[DenseColumn],
                 hidden_units: List[int],
                 activation_fn,
                 dropout: float,
                 batch_norm: bool = False,
                 name=None,
                 **kwargs):
        super(DNNModel, self).__init__(name=name, **kwargs)
        self.feature_columns = sorted(feature_columns, key=lambda column: column.name)
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.state_manager = _StateManagerImplV2(self, trainable=True)

        self.dense_layers = []
        self.dropout_layers = []
        self.batch_norm_layers = []
        for index, hidden_unit in enumerate(hidden_units):
            dense_layer = tf.keras.layers.Dense(units=hidden_unit, activation=activation_fn,
                                                name="dense_layer_%d" % index)
            self.dense_layers.append(dense_layer)
            if dropout:
                dropout_layer = tf.keras.layers.Dropout(rate=dropout, name="dropout_layer_%d" % index)
                self.dropout_layers.append(dropout_layer)
            if batch_norm:
                batch_norm_layer = tf.keras.layers.BatchNormalization(momentum=0.999,
                                                                      name="batch_norm_layer_%d" % index)
                self.batch_norm_layers.append(batch_norm_layer)

        self.logits_layer = tf.keras.layers.Dense(units=output_dim, activation=None, name="logits_layer")

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_columns
        config = {
            "output_dim": self.output_dim,
            "feature_columns": serialize_feature_columns(self.feature_columns),
            "hidden_units": tf.keras.utils.serialize_keras_object(self.hidden_units),
            "activation_fn": self.activation_fn,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm
        }
        base_config = super(DNNModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        for feature_column in self.feature_columns:
            feature_column.create_state(self.state_manager)

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        feature_column_tensors = []
        for feature_column in self.feature_columns:
            feature_tensor = feature_column.get_dense_tensor(transformation_cache, self.state_manager)
            feature_column_tensors.append(feature_tensor)
        output_tensor = tf.concat(feature_column_tensors, axis=1)
        for layer_index, dense_layer in enumerate(self.dense_layers):
            output_tensor = dense_layer(output_tensor)
            if self.dropout:
                output_tensor = self.dropout_layers[layer_index](output_tensor, training)
            if self.batch_norm:
                output_tensor = self.batch_norm_layers[layer_index](output_tensor, training)
        logits = self.logits_layer(output_tensor)
        predict = tf.keras.activations.sigmoid(logits)
        return predict


def main(_):
    import os
    titanic_train_data_path = os.path.abspath(__file__).replace("models/rank/nn/dnn.py", "data/titanic/train.csv")

    data = tf.data.experimental.make_csv_dataset(
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
        tf.feature_column.embedding_column(
            categorical_column=tf.feature_column.bucketized_column(
                source_column=tf.feature_column.numeric_column(key="Age"),
                boundaries=[1, 6, 14, 18, 30, 40, 50, 60, 70]),
            dimension=10),
        tf.feature_column.indicator_column(
            categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
                key="Pclass", vocabulary_list=[1, 2, 3])),
        tf.feature_column.indicator_column(
            categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
                key="Sex", vocabulary_list=["female", "male"]))
    ]

    model = DNNModel(output_dim=1, feature_columns=feature_columns, hidden_units=[128, 128, 128],
                     activation_fn=tf.keras.activations.relu, dropout=0, batch_norm=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=tf.keras.metrics.AUC())
    model.fit(x=data, epochs=10)


if __name__ == "__main__":
    app.run(main)
