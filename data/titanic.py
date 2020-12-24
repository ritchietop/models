import tensorflow as tf


class RawInputs:
    pclass_inputs = tf.keras.layers.Input(shape=(1,), name="Pclass", dtype=tf.int32)
    sex_inputs = tf.keras.layers.Input(shape=(1,), name="Sex", dtype=tf.string)
    age_inputs = tf.keras.layers.Input(shape=(1,), name="Age", dtype=tf.float32)
    sibsp_inputs = tf.keras.layers.Input(shape=(1,), name="SibSp", dtype=tf.int32)
    parch_inputs = tf.keras.layers.Input(shape=(1,), name="Parch", dtype=tf.int32)
    ticket_inputs = tf.keras.layers.Input(shape=(1,), name="Ticket", dtype=tf.string)
    fare_inputs = tf.keras.layers.Input(shape=(1,), name="Fare", dtype=tf.float32)
    cabin_inputs = tf.keras.layers.Input(shape=(1,), name="Cabin", dtype=tf.string)
    embarked_inputs = tf.keras.layers.Input(shape=(1,), name="Embarked", dtype=tf.string)


class PreProcess:
    pclass_preprocess = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=[1, 2, 3])
    sex_preprocess = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=["male", "female"])
    age_preprocess = tf.keras.layers.experimental.preprocessing.Discretization(bins=[1, 6, 14, 18, 30, 40, 50, 60, 70])
    sibsp_preprocess = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=list(range(9)), mask_value=None)
    parch_preprocess = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=list(range(6)), mask_value=None)
    ticket_preprocess = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=200)
    cabin_preprocess = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=20)
    embarked_preprocess = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=["S", "C", "Q"], mask_token=None)


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
