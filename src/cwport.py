import numpy as np

SPDR_WEIGHTS = {
    'XLB': 3.0,   # Materials
    'XLC': 11.0,  # Communication Services
    'XLE': 4.0,   # Energy
    'XLF': 11.0,  # Financials
    'XLI': 8.0,   # Industrials
    'XLK': 26.0,  # Information Technology
    'XLP': 7.0,   # Consumer Staples
    'XLRE': 3.0,  # Real Estate
    'XLU': 3.0,   # Utilities
    'XLY': 11.0,  # Consumer Discretionary
    'XLV': 15.0   # Health Care
}


def cw_port(cov, **ignore):
    weight_values = np.array(list(SPDR_WEIGHTS.values()))
    total = np.sum(weight_values)
    normalized_weights = weight_values / total
    return normalized_weights
