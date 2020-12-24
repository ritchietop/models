import tensorflow as tf
from absl import app
import os

from data.titanic import RawInputs, PreProcess, input_fn


def linear_model(output_dim):
    input_layers = [
        RawInputs.pclass_inputs, RawInputs.sex_inputs, RawInputs.age_inputs, RawInputs.sibsp_inputs,
        RawInputs.parch_inputs, RawInputs.ticket_inputs, RawInputs.cabin_inputs, RawInputs.embarked_inputs,
        RawInputs.fare_inputs
    ]
    preprocess_layers = [
        PreProcess.pclass_preprocess, PreProcess.sex_preprocess, PreProcess.age_preprocess, PreProcess.sibsp_preprocess,
        PreProcess.parch_preprocess, PreProcess.ticket_preprocess, PreProcess.cabin_preprocess,
        PreProcess.embarked_preprocess
    ]
    input_cast_layer = tf.keras.layers.Lambda(function=lambda tensor: tf.cast(tensor, dtype=tf.float32))
    inputs = [input_cast_layer(preprocess_layers[i](input_layers[i])) for i in range(len(preprocess_layers))]
    inputs.append(input_layers[-1])
    inputs = tf.keras.layers.Concatenate(axis=1)(inputs)
    predict = tf.keras.layers.Dense(units=output_dim, activation=tf.keras.activations.sigmoid)(inputs)
    model = tf.keras.models.Model(inputs=input_layers, outputs=predict)
    return model


def main(_):
    model = linear_model(output_dim=1)
    train_data_path = os.path.abspath(__file__).replace("example/rank/ml/lr.py", "data/titanic/train.csv")
    validate_data_path = os.path.abspath(__file__).replace("example/rank/ml/lr.py", "data/titanic/validate.csv")
    train_data = input_fn(train_data_path, batch_size=10)
    validate_data = input_fn(validate_data_path, batch_size=100)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.binary_crossentropy)
    model.fit(x=train_data, validation_data=validate_data, epochs=10)
    tf.keras.utils.plot_model(model, to_file="./lr.png")


if __name__ == "__main__":
    app.run(main)
