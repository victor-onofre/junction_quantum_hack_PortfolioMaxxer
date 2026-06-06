# junction_quantum_hack_PortfolioMaxxer

In financial terms, the script models a Binary Portfolio Selection problem.
Instead of deciding how much money to put into an asset, the model makes a
"Go/No-Go" decision on each sector.


1. The Objective Function f(x): "Risk-Adjusted Return"

The function f(x) represents the Total Score of a portfolio. A higher score
means a more desirable investment. It is composed of two financial metrics:

  - Expected Portfolio Return on Investment-ROI (The Linear Term):
      - sum(x_i * (return // 5))
      - Finance Meaning: This is the weighted sum of expected returns. The
        script looks at 6 sectors (e.g., Tech, Energy, Retail) and adds their
        projected profits to the total score. The "scaling" (dividing by 5)
        represents a discretized return class. Instead of looking at a 17.5%
        return, the model treats it as a "Category 3" return. This simplifies
        the decision-making process for the quantum algorithm.
  - Covariance Penalty / Diversification (The Quadratic Term):
      - -1 * x[0] * x[1]
      - Finance Meaning: This represents Risk Mitigation. In finance, if Asset 0
        and Asset 1 are highly correlated (e.g., two different oil companies),
        owning both doesn't diversify your risk—it doubles it. This term acts as
        a penalty for over-concentration. If the model picks both assets, the
        portfolio score is penalized by 1 point, encouraging the model to pick
        assets that move independently.

2. The Constraints: "Diversification & Structure"

The constraint sum(x) == 1 (mod 2) is the critical part of the model’s
structure:

  - Cardinality & Diversification:
      - Finance Meaning: In professional fund management, you rarely want a
        portfolio that is too small (no diversification) or too large (too much
        management overhead).
      - The "Odd-Number" Rule: This specific constraint requires the portfolio
        to contain an odd number of assets (1, 3, or 5).
          - It prevents a Null Portfolio (0 assets), which is financially
            useless.
          - It acts as a proxy for Minimum and Maximum Diversification. By
            forcing an odd number, the model must choose between a
            "concentrated" strategy (1 asset) or a "diversified" strategy (3
            or 5 assets).
  - Soft Constraints (Penalty Weights):
      - weight=3
      - Finance Meaning: This represents a Mandate. In finance, some rules are
        "soft" (targets) and some are "hard" (regulatory requirements). By
        setting the weight to 3, the script tells the algorithm that satisfying
        the portfolio structure (the odd-number rule) is three times more
        important than getting an extra point of ROI. This ensures the resulting
        portfolio is legally or structurally compliant before it tries to be
        profitable.

3. Financial Summary of the Script

If a fund manager were to describe the strategy inside this DQI-Kit script, they
would say:

"We are looking for a diversified portfolio of 1, 3, or 5 assets. We want to
maximize our total ROI categories, but we have a specific risk-avoidance rule
that forbids us from being too heavily invested in the specific correlation
between Asset 0 and Asset 1. We value the structural integrity of the portfolio
(the asset count) significantly more than marginal gains in return."

