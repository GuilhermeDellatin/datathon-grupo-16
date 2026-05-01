"""Re-export shim — código real vive em ``src.features.feature_engineering``.

Mantido para preservar imports legados (``from src.data.feature_engineering
import compute_features``) sem forçar churn nas chamadas existentes em
testes, DVC stages, Airflow DAGs e specs. Novo código deve importar a
partir de ``src.features.feature_engineering``.
"""

from src.features.feature_engineering import (
    FEATURE_SCHEMA,
    RAW_SCHEMA,
    compute_features,
    compute_technical_indicators,
    create_sequences,
    load_config,
    main,
    split_data,
    validate_feature_data,
    validate_raw_data,
)

__all__ = [
    "FEATURE_SCHEMA",
    "RAW_SCHEMA",
    "compute_features",
    "compute_technical_indicators",
    "create_sequences",
    "load_config",
    "main",
    "split_data",
    "validate_feature_data",
    "validate_raw_data",
]


if __name__ == "__main__":
    # Permite que `python -m src.data.feature_engineering` continue
    # funcionando (DVC, Airflow, Makefile usam essa entrada).
    main()
