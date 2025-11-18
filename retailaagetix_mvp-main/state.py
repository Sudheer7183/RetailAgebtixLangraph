# state.py
from typing import Dict, Any, TypedDict, Optional, List

class RetailState(TypedDict, total=False):
    run_id: str  # Added: For run-specific identification and logging
    category: str  # Added: To pass category (e.g., "perishables" or "bakery")
    ingestion: Dict[str, Any]  # Renamed from ingestion_output for consistency (match AGUI keys)
    analysis: Dict[str, Any]   # Renamed from analysis_output
    prediction: Dict[str, Any]
    decision: Dict[str, Any]
    execution: Dict[str, Any]
    prediction_agent: Any  # Added: To hold the trained PredictionAgent instance across nodes
    messages: Optional[List[Any]]  # Optional: If you want to keep message logging (from your node examples)