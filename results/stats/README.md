# Results / Statistics

This directory contains statstics of the results from the experiments/
evaluations conducted in the course of the thesis.

They are provided as CSV files and separated by dataset.

Each contains the following files:

- `..._ood_uncertainty_comparison.csv`: Comparison of mean uncertainty and
  standard deviation of different Out-of-Domain datasets compared to original
  data for all tested methods.
- `..._error_by_abstention.csv`: Maximum absolute errors and maximum 95th
  percentile absolute errors for different abstention rates (0 %, 10 %, ...,
  90 %) for different tested models. The respective uncertainty threshold is
  provided to each abstention rate, as well.
- `..._error_uncertainty_stats.csv`: Mean absolute error and mean uncertainty
  with corresponding median standard deviation for each tested method.
- `..._uncertainty_by_error_aucs.csv`: Area-under-Curves for uncertainty by
  error plots (*) for different tested methods.
- `..._uncertainty_by_error_reorder_ranks.csv`: The different tested statistics
  w.r.t. uncertainty error correlation (including the URD metric used in the
  thesis)

(*): This metric/ statistics from this CSV file have not been used in the
results or discussion of the thesis.