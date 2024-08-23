import numpy as np

SPDR_WEIGHTS = {
    'XLB': 3.0,   # Materials                  crunch -> 12%         smmv    0%
    'XLC': 11.0,  # Communication Services               14%                 6%
    'XLE': 4.0,   # Energy                               1%                  0%
    'XLF': 11.0,  # Financials                           9%                 21%
    'XLI': 8.0,   # Industrials                          8%                  1%
    'XLK': 26.0,  # Information Technology               12%                23%
    'XLP': 7.0,   # Consumer Staples                     4%                 21%
    'XLRE': 3.0,  # Real Estate                          9%                  1%
    'XLU': 3.0,   # Utilities                            2%                 10%
    'XLY': 11.0,  # Consumer Discretionary               25%                 6%
    'XLV': 15.0   # Health Care                          3%                 19%
}


def cw_port(cov, **ignore):
    weight_values = np.array(list(SPDR_WEIGHTS.values()))
    total = np.sum(weight_values)
    normalized_weights = weight_values / total
    return normalized_weights
