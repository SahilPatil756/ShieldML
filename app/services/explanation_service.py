"""
Explanation service for ShieldML.
Wraps the FraudExplainer to provide real-time top-feature explanations.
"""

from __future__ import annotations

import logging
from typing import Any
import pandas as pd
from shieldml.explainability.shap_explainer import FraudExplainer

logger = logging.getLogger(__name__)


class ExplanationService:
    def __init__(self, explainer: FraudExplainer | None = None):
        self.explainer = explainer

    def set_explainer(self, explainer: FraudExplainer) -> None:
        self.explainer = explainer

    def explain(self, X_scaled: pd.DataFrame, top_k: int = 5) -> list[dict[str, Any]]:
        if self.explainer is None:
            return []
        return self.explainer.explain_instance(X_scaled, top_k=top_k)


explanation_service = ExplanationService()
