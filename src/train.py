"""
Train the loan default prediction models.

We train two Spark ML classifiers on the engineered features and keep
whichever scores better on the holdout set:
  - Gradient-boosted trees (usually wins on tabular risk data)
  - Random forest (strong, fast baseline)

The winner is saved along with a small metrics.json so the allocation
engine and the notebook can report honest numbers. Train/test split is
random here; in production you'd split by origination date to avoid
lookahead.

Run:
    python -m src.train
"""

import json
import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

from src.config import from_env, spark_session


def evaluate(predictions, label_col="label"):
    """Compute AUC, accuracy and F1 on a predictions DataFrame."""
    auc = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction"
    ).evaluate(predictions)
    acc = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="accuracy"
    ).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)
    return {"auc": auc, "accuracy": acc, "f1": f1}


def main() -> None:
    settings = from_env()
    spark = spark_session(settings)

    data = spark.read.parquet(settings.features_dir)
    train, test = data.randomSplit(
        [settings.train_ratio, 1 - settings.train_ratio],
        seed=settings.random_seed,
    )
    print(f"Train: {train.count():,}  Test: {test.count():,}")

    candidates = {
        "gbt": GBTClassifier(
            labelCol="label",
            featuresCol="features",
            maxIter=settings.gbt_max_iter,
            maxDepth=settings.gbt_max_depth,
            seed=settings.random_seed,
        ),
        "random_forest": RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=settings.rf_num_trees,
            maxDepth=settings.rf_max_depth,
            seed=settings.random_seed,
        ),
    }

    results = {}
    best_name, best_score, best_model = None, -1.0, None
    for name, estimator in candidates.items():
        model = Pipeline(stages=[estimator]).fit(train)
        preds = model.transform(test)
        metrics = evaluate(preds)
        results[name] = metrics
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if metrics["auc"] > best_score:
            best_name, best_score, best_model = name, metrics["auc"], model

    best_model.write().overwrite().save(f"{settings.models_dir}/default_model")
    with open(os.path.join(settings.models_dir, "metrics.json"), "w") as f:
        json.dump({"best": best_name, "all": results}, f, indent=2)

    print(f"Best model: {best_name} (AUC={best_score:.4f})")
    print(f"Saved -> {settings.models_dir}/default_model")
    spark.stop()


if __name__ == "__main__":
    main()
