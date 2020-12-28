import tensorflow as tf
from models.rank.ml.poly2 import Poly2Model

titanic_train_data_path = "../../../data/titanic/train.csv"

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


pclass_column = tf.feature_column.categorical_column_with_vocabulary_list(key="Pclass", vocabulary_list=[1, 2, 3])
sex_column = tf.feature_column.categorical_column_with_vocabulary_list(key="Sex", vocabulary_list=["female", "male"])
age_column = tf.feature_column.bucketized_column(source_column=tf.feature_column.numeric_column(key="Age"),
                                                 boundaries=[1, 6, 14, 18, 30, 40, 50, 60, 70])

feature_columns = [pclass_column, sex_column, age_column]

cross_feature_columns = [
    tf.feature_column.crossed_column(keys=[sex_column, age_column], hash_bucket_size=100)
]

model = Poly2Model(columns=feature_columns, cross_columns=cross_feature_columns, l2_factor=0.5)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss=tf.keras.losses.binary_crossentropy)
model.fit(x=train_data, epochs=10)
