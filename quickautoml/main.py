from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from quickautoml.adapters import SKLearnModelsSupplier
from quickautoml.estimators import Classifier
from quickautoml.feature_engineering import PandasFeatureEngineer
from quickautoml.hyperparameter_optimizer import OptunaHyperparamsOptimizer
from quickautoml.preprocessors import PandasDataPreprocessor


def split_dataframe(dataframe, reference_col: str = 'class'):
  return dataframe.drop(reference_col, axis=1), dataframe[reference_col]


def make_classifier():
  data_preprocessor = PandasDataPreprocessor()
  feature_engineer = PandasFeatureEngineer()
  hyperparameter_optimizer = OptunaHyperparamsOptimizer('accuracy')
  models_supplier = SKLearnModelsSupplier()
  return Classifier(data_preprocessor, feature_engineer, models_supplier, hyperparameter_optimizer)


if __name__ == '__main__':
  from pandas import read_csv
  cls = make_classifier()
  df = read_csv("../datasets/apcd_permissions_limpo.csv", index_col=0)
  X, y = split_dataframe(df)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.75)
  cls.fit(X_train, y_train)
  predictions = cls.predict(X_test)
  print(f"recall: {recall_score(y_test, predictions)}")
