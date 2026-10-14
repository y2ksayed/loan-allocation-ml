"""
Capital allocation engine.

Given a pool of loan applications and a fixed amount of capital, decide
how much to fund per loan. The strategy is deliberately simple and
explainable — the kind of thing you'd actually defend in a risk review:

  1. Score every application with the trained default model.
  2. Estimate expected return = interest_rate * (1 - p_default)
     minus a loss-given-default haircut on the principal at risk.
  3. Drop loans below the minimum expected return.
  4. Fund in descending expected-return order, capped per loan,
     until capital runs out.

Run:
    python -m src.allocate --capital 10000000
"""

import argparse

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import from_env, spark_session
from src.features import add_derived_columns, build_feature_pipeline


def score_applications(spark, settings):
    """Load raw applications, engineer features, and add default probabilities."""
    raw = spark.read.parquet(f"{settings.raw_dir}/applications.parquet")
    df = add_derived_columns(raw)

    featurizer = PipelineModel.load(f"{settings.models_dir}/featurizer")
    model = PipelineModel.load(f"{settings.models_dir}/default_model")

    featured = featurizer.transform(df)
    scored = model.transform(featured)
    # probability column is a vector [P(no default), P(default)]
    return scored.withColumn("p_default", F.col("probability")[1])


def expected_return(df, lgd: float = 0.45):
    """Add expected annual return per dollar of principal."""
    return df.withColumn(
        "expected_return",
        F.col("interest_rate") * (1 - F.col("p_default"))
        - lgd * F.col("p_default"),
    )


def allocate(df, capital: float, settings):
    """Greedy allocation: best expected return first, with a per-loan cap."""
    eligible = df.filter(F.col("expected_return") >= settings.min_expected_return)
    ranked = eligible.withColumn(
        "rank", F.row_number().over(Window.orderBy(F.col("expected_return").desc()))
    )
    # Fund each loan up to the cap, in rank order, until capital is spent.
    # We do this with a running total so it's a single pass.
    with_running = ranked.withColumn(
        "cum_cap",
        F.sum(
            F.least(F.col("loan_amount"),
                    F.lit(settings.max_exposure_per_loan))
        ).over(Window.orderBy("rank").rowsBetween(Window.unboundedPreceding, 0)),
    )
    funded = with_running.filter(F.col("cum_cap") <= capital).withColumn(
        "allocated",
        F.least(F.col("loan_amount"), F.lit(settings.max_exposure_per_loan)),
    )
    return funded.select(
        "application_id",
        "loan_amount",
        "interest_rate",
        "p_default",
        "expected_return",
        "allocated",
    )


def main() -> None:
    settings = from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=settings.total_capital)
    parser.add_argument("--lgd", type=float, default=0.45,
                        help="loss given default, fraction of principal")
    args = parser.parse_args()

    settings.ensure_dirs()
    spark = spark_session(settings)

    scored = expected_return(score_applications(spark, settings), lgd=args.lgd)
    portfolio = allocate(scored, args.capital, settings)

    out = f"{settings.output_dir}/allocation.parquet"
    portfolio.write.mode("overwrite").parquet(out)

    stats = portfolio.agg(
        F.count("*").alias("n_loans"),
        F.sum("allocated").alias("capital_deployed"),
        F.avg("expected_return").alias("avg_expected_return"),
        F.avg("p_default").alias("avg_p_default"),
    ).collect()[0]
    print(f"Allocated to {stats['n_loans']} loans")
    print(f"Capital deployed: ${stats['capital_deployed']:,.0f} of ${args.capital:,.0f}")
    print(f"Avg expected return: {stats['avg_expected_return']:.3%}")
    print(f"Avg default prob:    {stats['avg_p_default']:.3%}")
    print(f"Wrote -> {out}")
    spark.stop()


if __name__ == "__main__":
    main()
