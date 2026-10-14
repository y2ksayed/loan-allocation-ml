"""
Shared configuration for the loan allocation ML project.

Everything environment-specific lives here so the rest of the code
stays clean: paths, Spark settings, feature definitions, and model
hyperparameters. Override anything with environment variables when
you move this to a cluster.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Where data lands
    raw_dir: str = "data/raw"
    features_dir: str = "data/features"
    models_dir: str = "models"
    output_dir: str = "data/output"

    # Synthetic data generation
    n_applications: int = 50_000
    default_base_rate: float = 0.12
    random_seed: int = 42

    # Spark session
    spark_app_name: str = "loan-allocation-ml"
    spark_master: str = "local[*]"
    spark_shuffle_partitions: int = 8

    # Feature engineering
    credit_score_buckets: list = field(
        default_factory=lambda: [300, 580, 670, 740, 800, 850]
    )

    # Model training
    train_ratio: float = 0.8
    gbt_max_iter: int = 60
    gbt_max_depth: int = 6
    rf_num_trees: int = 100
    rf_max_depth: int = 8

    # Allocation engine
    total_capital: float = 10_000_000.0
    max_exposure_per_loan: float = 250_000.0
    min_expected_return: float = 0.04  # skip loans below 4% expected return

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.features_dir, self.models_dir, self.output_dir):
            os.makedirs(d, exist_ok=True)


def from_env() -> Settings:
    """Build settings, letting a few env vars override the defaults."""
    s = Settings()
    s.spark_master = os.getenv("SPARK_MASTER", s.spark_master)
    s.total_capital = float(os.getenv("TOTAL_CAPITAL", s.total_capital))
    s.n_applications = int(os.getenv("N_APPLICATIONS", s.n_applications))
    return s


def spark_session(settings: Settings):
    """Create a local Spark session with sane defaults for this project."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(settings.spark_app_name)
        .master(settings.spark_master)
        .config("spark.sql.shuffle.partitions", settings.spark_shuffle_partitions)
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


if __name__ == "__main__":
    s = from_env()
    s.ensure_dirs()
    print("Directories ready:")
    for d in (s.raw_dir, s.features_dir, s.models_dir, s.output_dir):
        print(f"  - {d}")
