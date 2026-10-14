"""
Spark feature engineering for loan applications.

Turns the raw application table into a model-ready feature matrix:
bucketed credit scores, one-hot home ownership, derived ratios, and a
single assembled feature vector. We persist the fitted transformers so
inference reuses exactly the same encoding — a classic source of
train/serve skew if you skip it.

Run:
    python -m src.features
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Bucketizer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import functions as F

from src.config import from_env, spark_session


def build_feature_pipeline(settings) -> Pipeline:
    """Assemble the preprocessing pipeline (fitted later on training data)."""
    # Credit score -> risk band buckets, e.g. subprime / near-prime / prime
    credit_bucketizer = Bucketizer(
        splits=settings.credit_score_buckets,
        inputCol="credit_score",
        outputCol="credit_band",
    )
    home_indexer = StringIndexer(
        inputCol="home_ownership", outputCol="home_idx", handleInvalid="keep"
    )
    home_encoder = OneHotEncoder(
        inputCols=["home_idx"], outputCols=["home_vec"]
    )
    assembler = VectorAssembler(
        inputCols=[
            "credit_band",
            "annual_income",
            "dti",
            "loan_amount",
            "term_months",
            "employment_years",
            "interest_rate",
            "loan_to_income",
            "monthly_payment_ratio",
            "home_vec",
        ],
        outputCol="features",
        handleInvalid="skip",
    )
    return Pipeline(
        stages=[credit_bucketizer, home_indexer, home_encoder, assembler]
    )


def add_derived_columns(df):
    """Add ratios the raw table doesn't have but models benefit from."""
    return (
        df.withColumn("loan_to_income", F.col("loan_amount") / F.col("annual_income"))
        # Rough monthly payment share of income — a useful affordability signal
        .withColumn(
            "monthly_payment_ratio",
            (F.col("loan_amount") / F.col("term_months"))
            / (F.col("annual_income") / 12),
        )
        .withColumn("label", F.col("defaulted").cast("double"))
    )


def main() -> None:
    settings = from_env()
    settings.ensure_dirs()
    spark = spark_session(settings)

    raw = spark.read.parquet(f"{settings.raw_dir}/applications.parquet")
    df = add_derived_columns(raw)

    pipeline = build_feature_pipeline(settings)
    model = pipeline.fit(df)
    featured = model.transform(df)

    featured.select("application_id", "features", "label").write.mode(
        "overwrite"
    ).parquet(settings.features_dir)
    model.write().overwrite().save(f"{settings.models_dir}/featurizer")

    print(f"Feature rows: {featured.count():,}")
    print(f"Saved features -> {settings.features_dir}")
    print(f"Saved featurizer -> {settings.models_dir}/featurizer")
    spark.stop()


if __name__ == "__main__":
    main()
