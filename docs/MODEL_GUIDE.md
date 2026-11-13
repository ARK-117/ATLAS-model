# Model and independent evaluation

## Mathematical model

For each completed daily bar, the model uses one-session close return, close divided by the preceding
five-bar high minus one, and log(1 + current volume / preceding five-bar mean volume).
Missing history and zero-volume observations do not produce a training observation.

Training computes feature means and population standard deviations from the training rows alone.
With standardized feature matrix Z and demeaned return y, the coefficients solve
(Z^T Z / n + alpha I) beta = Z^T y / n. Ordinary least squares uses alpha = 0; ridge uses alpha = 0.10.
Constant columns receive a zero coefficient and a scale of one. The intercept is the mean training return.
Prediction is intercept + sum(beta_j * (x_j - mean_j) / scale_j).

The experiment compares 5- and 20-session raw close-to-close targets. A label is admitted only after its
target date and strictly before the validation boundary. Three expanding validation folds occupy the
later half of development. Development results select the candidate; holdout scoring uses its frozen fit.
Regression signals use an illustrative 1% return hurdle. The alternative baseline compares completed
closes with the preceding five-bar range and a 1.5 relative-volume threshold.

## Execution and evaluation

The shared paper engine models next-session eligibility, limit prices, adverse spread/slippage,
configured lots, volume participation, dated fees, cash/inventory reservations, and explicit settlement dates.
Metrics include marked equity, net return, drawdown, turnover, exposure, fees, and fill counts.
Base, higher-cost, and lower-liquidity scenarios are compared with flat cash and a passive benchmark.
The passive benchmark excludes trading costs. Taxes and complete corporate-action accounting are outside
this experiment. Daily bars cannot establish intraday liquidity, execution queue position, or actual fills.

The bundled demo uses invented symbols, prices, weekday calendars and assumptions.
It verifies calculations, and is not a NEPSE historical-performance result.
Live data fetching, dashboard state, private portfolios and account access are absent from this package.

## Independent tests

1. Run the complete quality command before making changes and retain the commit SHA with the result.
2. Inspect the known-line OLS/ridge tests and numerical edge cases in test_regression.py.
3. Check that changing future/holdout bars cannot change prior decisions or model selection.
4. Use a separate, permitted dataset and choose development/holdout windows before tuning.
5. Run all cost/liquidity scenarios and compare against cash and the passive benchmark.
6. Record data revisions, fees, missing sessions, corporate actions, and failed assumptions alongside returns.
7. Submit proposed changes and reproducible failures in pull requests.

Custom specifications supplied to --spec use bars, benchmark, calendar, settlement_calendar, risk and
terms paths relative to the specification. See load_backtest_spec in atlas/backtest_cli.py for the complete
contract. History declarations require verified point-in-time, action-free, constant-rule data.
They must not be set merely to get past validation.
