import tensorflow as tf

titanic_data_path = "/data/titanic/train.csv"

data = tf.data.experimental.make_csv_dataset(
    file_pattern=titanic_data_path,
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

from models.rank.ml.lr import LinearModel
from models.rank.ml.fm import FMModel
from models.rank.ml.ffm import FFMModel
from models.rank.ml.mlr import MLRModel

model = LinearModel(
    columns=[
        tf.feature_column.categorical_column_with_vocabulary_list(key="Pclass", vocabulary_list=[1, 2, 3]),
        tf.feature_column.embedding_column(
            categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
                key="Sex", vocabulary_list=["female", "male"]),
            dimension=10)
    ])

model = FMModel(
    latent_dim=10,
    columns=[
        tf.feature_column.categorical_column_with_vocabulary_list(key="Pclass", vocabulary_list=[1, 2, 3]),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="Sex", vocabulary_list=["female", "male"]),
        # tf.feature_column.embedding_column(
        #     categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        #         key="Sex", vocabulary_list=["female", "male"]),
        #     dimension=10)
    ])

model = FFMModel(
    latent_dim=10,
    columns=[
        tf.feature_column.categorical_column_with_vocabulary_list(key="Pclass", vocabulary_list=[1, 2, 3]),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="Sex", vocabulary_list=["female", "male"]),
    ]
)

model = MLRModel(
    split_count=10,
    columns=[
        tf.feature_column.categorical_column_with_vocabulary_list(key="Pclass", vocabulary_list=[1, 2, 3]),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="Sex", vocabulary_list=["female", "male"]),
    ]
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mean_squared_error)
# print(model.trainable_variables)
model.fit(data, epochs=10)

# for features, label in data:
#     print(features)
#     print(label)
#     break
