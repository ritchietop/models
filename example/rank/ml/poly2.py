import tensorflow as tf
from absl import app
import os


def poly2_model(output_dim):
    # raw inputs
    pclass_input = tf.keras.layers.Input(shape=(1,), name="Pclass", dtype=tf.int32)
    sex_input = tf.keras.layers.Input(shape=(1,), name="Sex", dtype=tf.string)
    age_input = tf.keras.layers.Input(shape=(1,), name="Age", dtype=tf.float32)
    sibsp_input = tf.keras.layers.Input(shape=(1,), name="SibSp", dtype=tf.int32)
    parch_input = tf.keras.layers.Input(shape=(1,), name="Parch", dtype=tf.int32)
    ticket_input = tf.keras.layers.Input(shape=(1,), name="Ticket", dtype=tf.string)
    fare_input = tf.keras.layers.Input(shape=(1,), name="Fare", dtype=tf.float32)
    cabin_input = tf.keras.layers.Input(shape=(1,), name="Cabin", dtype=tf.string)
    embarked_input = tf.keras.layers.Input(shape=(1,), name="Embarked", dtype=tf.string)

    # linear inputs
    pclass_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=3)(
        tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=[1, 2, 3], mask_value=None)(pclass_input))
    sex_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=3)(
        tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=["male", "female"])(sex_input))
    age_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=10)(
        tf.keras.layers.experimental.preprocessing.Discretization(
            bins=[1, 6, 14, 18, 30, 40, 50, 60, 70])(age_input))
    sibsp_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=9)(
        tf.keras.layers.experimental.preprocessing.IntegerLookup(
            vocabulary=list(range(9)), mask_value=None)(sibsp_input))
    parch_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=6)(
        tf.keras.layers.experimental.preprocessing.IntegerLookup(
            vocabulary=list(range(6)), mask_value=None)(parch_input))
    ticket_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=200)(
        tf.keras.layers.experimental.preprocessing.Hashing(num_bins=200)(ticket_input))
    cabin_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=20)(
        tf.keras.layers.experimental.preprocessing.Hashing(num_bins=20)(cabin_input))
    embarked_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=3)(
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=["S", "C", "Q"], mask_token=None)(embarked_input))

    # cross inputs
    sex_x_age_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=30)(
        tf.keras.layers.experimental.preprocessing.Hashing(num_bins=30)(
            tf.keras.layers.experimental.preprocessing.CategoryCrossing(depth=2)([sex_input, age_input])))
    sex_x_cabin_input_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=60)(
        tf.keras.layers.experimental.preprocessing.Hashing(num_bins=60)(
            tf.keras.layers.experimental.preprocessing.CategoryCrossing(depth=2)([sex_input_layer, cabin_input])))

    inputs = tf.keras.layers.Concatenate(axis=1)([
        pclass_input_layer, sex_input_layer, age_input_layer, sibsp_input_layer, parch_input_layer,
        ticket_input_layer, cabin_input_layer, embarked_input_layer, fare_input, sex_x_age_input_layer,
        sex_x_cabin_input_layer
    ])
    predict = tf.keras.layers.Dense(units=output_dim, activation=tf.keras.activations.sigmoid)(inputs)
    model = tf.keras.models.Model(inputs=[
        pclass_input, sex_input, age_input, sibsp_input, parch_input, ticket_input, fare_input, cabin_input,
        embarked_input
    ], outputs=predict)
    return model


def input_fn(path, batch_size, num_epochs=1, shuffle_buffer_size=None):
    return tf.data.experimental.make_csv_dataset(
        file_pattern=path,
        batch_size=batch_size,
        column_names=["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare",
                      "Cabin", "Embarked"],
        column_defaults=[0, 0, "", "", 0.0, 0, 0, "", 0.0, "", ""],
        label_name="Survived",
        select_columns=["Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin",
                        "Embarked"],
        field_delim=",",
        use_quote_delim=True,
        na_value="null",
        header=True,
        num_epochs=num_epochs,
        shuffle=bool(shuffle_buffer_size),
        shuffle_buffer_size=shuffle_buffer_size,
        num_rows_for_inference=0,
        ignore_errors=False)


def main(_):
    model = poly2_model(output_dim=1)
    train_data_path = os.path.abspath(__file__).replace("example/rank/ml/poly2.py", "data/titanic/train.csv")
    validate_data_path = os.path.abspath(__file__).replace("example/rank/ml/poly2.py", "data/titanic/validate.csv")
    train_data = input_fn(train_data_path, batch_size=10)
    validate_data = input_fn(validate_data_path, batch_size=100)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.binary_crossentropy)
    model.fit(x=train_data, validation_data=validate_data, epochs=10)
    tf.keras.utils.plot_model(model, to_file="./poly2.png")


if __name__ == "__main__":
    app.run(main)
