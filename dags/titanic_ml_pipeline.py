from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

DATA_PATH = '/opt/airflow/data/titanic.csv'

@dag(
    start_date=datetime(2026, 3, 11),
    schedule=None,
    catchup=False,
    tags=['titanic'],
    params={
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2
    }
)
def titanic_ml_pipeline():

    @task
    def ingest():
        df = pd.read_csv(DATA_PATH)
        output_path = '/opt/airflow/data/raw.csv'
        df.to_csv(output_path, index=False)
        return output_path

    @task
    def validate(path):
        df = pd.read_csv(path)
        df = df.dropna(subset=['Survived'])
        output_path = '/opt/airflow/data/validated.csv'
        df.to_csv(output_path, index=False)
        return output_path

    @task
    def impute_age(path):
        df = pd.read_csv(path)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        output_path = '/opt/airflow/data/imputed.csv'
        df.to_csv(output_path, index=False)
        return output_path

    @task
    def impute_embarked(path):
        df = pd.read_csv(path)
        df['Embarked'] = df['Embarked'].fillna('S')
        output_path = '/opt/airflow/data/imputed_embarked.csv'
        df.to_csv(output_path, index=False)
        return output_path

    @task
    def features(path):
        df = pd.read_csv(path)
        df['FamilySize'] = df['SibSp'] + df['Parch']
        output_path = '/opt/airflow/data/features.csv'
        df.to_csv(output_path, index=False)
        return output_path

    @task
    def encode(path):
        df = pd.read_csv(path)
        enc = OneHotEncoder(sparse_output=False)
        encoded_cols = enc.fit_transform(df[['Sex', 'Embarked']])
        encoded_df = pd.DataFrame(encoded_cols, columns=enc.get_feature_names_out())
        df = df.drop(columns=['Sex', 'Embarked'])
        df = pd.concat([df, encoded_df], axis=1)
        output_path = '/opt/airflow/data/encoded.csv'
        df.to_csv(output_path, index=False)
        return output_path

    @task
    def train(path, **context):
        import joblib
        params = context['params']
        n_estimators = params['n_estimators']
        max_depth = params['max_depth']
        min_samples_split = params['min_samples_split']

        df = pd.read_csv(path)
        X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        model.fit(X_train, y_train)
        output_path = '/opt/airflow/data/model.pkl'
        joblib.dump((model, X_test, y_test, params), output_path)
        return output_path

    @task
    def evaluate(path):
        import joblib
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        model, X_test, y_test, params = joblib.load(path)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        return acc

    @task
    def branch(acc):
        if acc > 0.75:
            return 'register_model'
        else:
            return 'reject_model'

    @task
    def register_model(path):
        import joblib
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        model, X_test, y_test, params = joblib.load(path)
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, preds_proba)

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("titanic")
        with mlflow.start_run(run_name=f"run_n{params['n_estimators']}_d{params['max_depth']}_s{params['min_samples_split']}"):
            # log hyperparameters
            mlflow.log_param("n_estimators", params['n_estimators'])
            mlflow.log_param("max_depth", params['max_depth'])
            mlflow.log_param("min_samples_split", params['min_samples_split'])

            # log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)

            mlflow.sklearn.log_model(
                model,
                artifact_path="titanic_rf_model",
                registered_model_name="titanic_rf_model"
            )
        return "Model registered"

    @task
    def reject_model():
        print("Model rejected")
        return "Rejected"

    @task
    def done():
        print("Pipeline completed")

    raw = ingest()
    validated = validate(raw)
    age_imputed = impute_age(validated)
    embarked_imputed = impute_embarked(age_imputed)
    featured = features(embarked_imputed)
    encoded = encode(featured)
    model_path = train(encoded)
    acc = evaluate(model_path)
    branch_task = branch(acc)

    registered = register_model(model_path)
    rejected = reject_model()
    done_task = done()

    branch_task >> registered >> done_task
    branch_task >> rejected >> done_task


dag_instance = titanic_ml_pipeline()