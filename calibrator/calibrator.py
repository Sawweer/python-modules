import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from joblib import Parallel, delayed

rating_labels = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
rating_pds = [
    0.000716273844667,
    0.002113681341366,
    0.004123666072761,
    0.008045026251994,
    0.015695365786964,
    0.030620721358806,
    0.059739198771169,
    0.116547609313422,
    0.227377425812238,
    0.443599779296768,
]


class Calibrator:
    """
    Randomized score binning
    """

    def __init__(
        self,
        scores,
        rating_labels,
        rating_pds,
        default_labels,
        verbose=True
    ):
        """
        Initialize with credit scores, rating labels, PDs per rating, and default labels.

        Parameters:
        -----------
        scores : array-like
            Credit scores for each observation
        rating_labels : list
            List of rating labels (e.g., ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D'])
        rating_pds : array-like (length 10)
            PD for each rating bucket
        default_labels : array-like
            Binary labels (1=default, 0=non-default) for each observation
        verbose : bool
            If True, print progress information
        """
        self.scores = np.array(scores)
        self.rating_labels = rating_labels
        self.rating_pds = np.array(rating_pds)
        self.default_labels = np.array(default_labels)
        self.n_ratings = len(rating_labels)
        self.verbose = verbose

        # Validate inputs
        assert (
            len(self.rating_pds) == self.n_ratings
        ), "PD array length must match number of ratings"
        assert len(self.scores) == len(
            self.default_labels
        ), "Scores and labels must have same length"

    def create_bins_from_thresholds(self, thresholds):
        """
        Create bins based on score thresholds.
        """
        thresholds = np.array(sorted(thresholds, reverse=True))
        bins = np.zeros(len(self.scores), dtype=int)

        for i, score in enumerate(self.scores):
            rating_idx = self.n_ratings - 1
            for j, threshold in enumerate(thresholds):
                if score >= threshold:
                    rating_idx = j
                    break
            bins[i] = rating_idx

        return bins

    def calculate_gini_from_pd(self, bins):
        """
        Calculate Gini coefficient using PD mapped to bins.
        """
        mapped_pds = self.rating_pds[bins]
        auc = roc_auc_score(self.default_labels, mapped_pds)
        gini = 2 * auc - 1
        return {"auc": auc, "ar": gini}

    def calculate_implied_pd(self, bins):
        bins_pd = self.rating_pds[bins]
        return bins_pd.mean()

    def binomial_test_bins(self, bins, alpha=0.05):
        """
        Conduct 1-tail binomial test on all bins.
        """
        bin_stats = []
        for rating_idx in range(self.n_ratings):
            mask = bins == rating_idx
            if not mask.any():
                continue
            n = mask.sum()
            observed_defaults = self.default_labels[mask].sum()
            expected_pd = self.rating_pds[rating_idx]
            observed_rate = observed_defaults / n
            pvalue = stats.binomtest(
                int(observed_defaults), int(n), expected_pd, alternative="greater"
            ).pvalue
            bin_stats.append(
                {
                    "rating": self.rating_labels[rating_idx],
                    "rating_idx": rating_idx,
                    "count": int(n),
                    "observed_defaults": int(observed_defaults),
                    "expected_pd": expected_pd,
                    "observed_rate": observed_rate,
                    "pvalue": pvalue,
                    "significant": pvalue <= alpha,
                }
            )
        results_df = pd.DataFrame(bin_stats)
        n_significant = results_df["significant"].sum() if len(
            results_df) > 0 else 0
        return results_df, int(n_significant)

    def check_reversals(self, bins):
        """
        Check for default rate reversals.
        """
        bin_summary = []
        for rating_idx in range(self.n_ratings):
            mask = bins == rating_idx
            if not mask.any():
                continue
            n = mask.sum()
            defaults = self.default_labels[mask].sum()
            default_rate = defaults / n
            bin_summary.append(
                {
                    "rating": self.rating_labels[rating_idx],
                    "rating_idx": rating_idx,
                    "count": int(n),
                    "defaults": int(defaults),
                    "default_rate": default_rate,
                    "expected_pd": self.rating_pds[rating_idx],
                }
            )
        summary_df = pd.DataFrame(bin_summary)
        reversals = []
        for i in range(len(summary_df) - 1):
            current_rate = summary_df.iloc[i]["default_rate"]
            next_rate = summary_df.iloc[i + 1]["default_rate"]
            if current_rate > next_rate:
                reversals.append(
                    {
                        "rating_better": summary_df.iloc[i]["rating"],
                        "rating_worse": summary_df.iloc[i + 1]["rating"],
                        "rate_better": current_rate,
                        "rate_worse": next_rate,
                        "difference": current_rate - next_rate,
                    }
                )
        return {
            "has_reversals": len(reversals) > 0,
            "n_reversals": len(reversals),
            "reversals": reversals,
            "bin_summary": summary_df,
        }

    def homogeneity_test(self, bins):
        """
        Test homogeneity across rating bins.
        """
        validation = []
        for rating_idx in range(self.n_ratings):
            mask = bins == rating_idx
            if not mask.any():
                continue
            total = mask.sum()
            defaults = self.default_labels[mask].sum()
            dr = defaults / total
            homo_lower = stats.binom.ppf(0.025, total, dr) / total
            homo_upper = stats.binom.ppf(0.975, total, dr) / total
            validation.append(
                {
                    "rating": self.rating_labels[rating_idx],
                    "rating_idx": rating_idx,
                    "total": int(total),
                    "defaults": int(defaults),
                    "dr": dr,
                    "homo_lower": homo_lower,
                    "homo_upper": homo_upper,
                }
            )
        validation_df = pd.DataFrame(validation)
        if len(validation_df) == 0:
            return {
                "n_homogeneity_fails": 0,
                "validation_df": validation_df,
                "fails": [],
            }
        validation_df = validation_df.sort_values(
            "rating_idx", ascending=False
        ).reset_index(drop=True)
        homogeneity_fails = []
        for i in range(len(validation_df) - 1):
            current_lower = validation_df.iloc[i]["homo_lower"]
            next_upper = validation_df.iloc[i + 1]["homo_upper"]
            if current_lower <= next_upper:
                homogeneity_fails.append(
                    {
                        "rating_current": validation_df.iloc[i]["rating"],
                        "rating_next": validation_df.iloc[i + 1]["rating"],
                        "current_lower": current_lower,
                        "next_upper": next_upper,
                        "overlap": next_upper - current_lower,
                    }
                )
        return {
            "n_homogeneity_fails": len(homogeneity_fails),
            "validation_df": validation_df,
            "fails": homogeneity_fails,
        }

    def check_bell_curve(self, proportions):
        """
        Check if the distribution of bin proportions follows a bell curve shape.

        Parameters:
        -----------
        proportions : pd.Series
            Proportions of the bins after sorting.

        Returns:
        --------
        bool
            True if the distribution follows a bell curve, False otherwise.
        """
        # Here we check for a simple criterion: the proportions should form a peak in the middle
        mid = len(proportions) // 2
        left = proportions.iloc[:mid]
        right = proportions.iloc[mid:]

        # For a bell curve, we expect the proportions to be highest in the middle
        left_sum = left.sum()
        right_sum = right.sum()

        # Simple check: if the left side's sum is greater than the right side's sum, it's a bell curve
        return left_sum < right_sum

    def _generate_beta_thresholds(self, alpha_param, beta_param, score_range):
        """
        Generate thresholds using Beta distribution transformation.

        Parameters:
        -----------
        alpha_param : float
            Alpha parameter for Beta distribution
        beta_param : float
            Beta parameter for Beta distribution
        score_range : tuple
            (min_score, max_score) for scaling thresholds

        Returns:
        --------
        array of threshold values
        """
        min_score, max_score = score_range

        # Create evenly spaced points from 0.1 to 0.9 (n_ratings - 1 points)
        x = np.linspace(0.1, 0.9, self.n_ratings - 1)

        # Transform through Beta CDF (ppf gives us the inverse)
        # This maps uniform spacing to Beta-distributed spacing
        proportions = stats.beta.cdf(x, alpha_param, beta_param)

        # Scale to score range
        thresholds = min_score + proportions * (max_score - min_score)

        # Sort descending (highest threshold first for best rating)
        thresholds = np.sort(thresholds)[::-1]

        return thresholds

    def simulate_thresholds_beta_parallel(
        self,
        alpha_range=(0.5, 5),
        beta_range=(0.5, 5),
        alpha_step=0.5,
        beta_step=0.5,
        score_range=None,
        n_jobs=-1,
    ):
        """
        Simulate threshold scenarios using Beta distribution with parallel processing.

        Parameters:
        -----------
        alpha_range : tuple
            (min_alpha, max_alpha) for Beta distribution alpha parameter
        beta_range : tuple
            (min_beta, max_beta) for Beta distribution beta parameter
        alpha_step : float
            Step size for alpha parameter grid
        beta_step : float
            Step size for beta parameter grid
        score_range : tuple
            (min_score, max_score) for generating thresholds. If None, uses data range
        n_jobs : int
            Number of parallel jobs. -1 uses all available cores

        Returns:
        --------
        DataFrame with results from all simulations
        """
        from joblib import Parallel, delayed
        from tqdm import tqdm

        if score_range is None:
            min_score, max_score = self.scores.min(), self.scores.max()
        else:
            min_score, max_score = score_range

        alphas = np.arange(
            alpha_range[0], alpha_range[1] + alpha_step, alpha_step)
        betas = np.arange(beta_range[0], beta_range[1] + beta_step, beta_step)
        param_combinations = [(a, b) for a in alphas for b in betas]

        def process_single(sim_id, params):
            alpha_param, beta_param = params
            thresholds = self._generate_beta_thresholds(
                alpha_param, beta_param, (min_score, max_score)
            )
            bins = self.create_bins_from_thresholds(thresholds)
            implied_pd = self.calculate_implied_pd(bins)
            gini_results = self.calculate_gini_from_pd(bins)
            _, n_significant = self.binomial_test_bins(bins)
            reversal_results = self.check_reversals(bins)
            homogeneity_results = self.homogeneity_test(bins)
            proportions = pd.Series(bins).value_counts(normalize=True)

            return {
                "simulation_id": sim_id,
                "alpha": alpha_param,
                "beta": beta_param,
                "thresholds": ",".join([f"{t:.1f}" for t in thresholds]),
                "implied_pd": implied_pd,
                "auc": gini_results["auc"],
                "ar": gini_results["ar"],
                "n_binomial_fail": n_significant,
                "n_reversal_fail": reversal_results["n_reversals"],
                "n_homogeneity_fail": homogeneity_results["n_homogeneity_fails"],
                "max_concentration": proportions.iloc[0],
                "min_concentration": (
                    proportions.iloc[-1] if len(
                        proportions) == self.n_ratings else 0
                ),
            }

        print(f"Starting Beta threshold simulation")
        print(
            f"  Alpha range: {alpha_range}, step: {alpha_step} ({len(alphas)} values)"
        )
        print(
            f"  Beta range: {beta_range}, step: {beta_step} ({len(betas)} values)")
        print(f"  Total iterations: {len(param_combinations)}")
        print(f"  Parallel jobs: {n_jobs if n_jobs != -1 else 'all cores'}")
        print("-" * 50)

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single)(sim_id, params)
            for sim_id, params in tqdm(
                enumerate(param_combinations),
                total=len(param_combinations),
                desc="Simulating Beta thresholds",
            )
        )

        self.beta_simulation_results = (
            pd.DataFrame(results).sort_values(
                "simulation_id").reset_index(drop=True)
        )
        return self.beta_simulation_results

    def simulate_thresholds_uniform_parallel(
        self,
        n_simulations=100,
        score_range=None,
        n_jobs=-1,
        random_seed=None,
    ):
        """
        Simulate multiple binning threshold scenarios using random sampling with parallel processing.

        Parameters:
        -----------
        n_simulations : int
            Number of threshold combinations to simulate
        score_range : tuple
            (min_score, max_score) for generating thresholds. If None, uses data range
        n_jobs : int
            Number of parallel jobs. -1 uses all available cores
        random_seed : int, optional
            Random seed for reproducibility

        Returns:
        --------
        DataFrame with results from all simulations
        """
        from joblib import Parallel, delayed
        from tqdm import tqdm

        if score_range is None:
            min_score, max_score = self.scores.min(), self.scores.max()
        else:
            min_score, max_score = score_range

        if random_seed is not None:
            np.random.seed(random_seed)

        all_thresholds = [
            np.sort(np.random.uniform(
                min_score, max_score, self.n_ratings - 1))[::-1]
            for _ in range(n_simulations)
        ]

        def process_single(sim_id, thresholds):
            bins = self.create_bins_from_thresholds(thresholds)
            implied_pd = self.calculate_implied_pd(bins)
            gini_results = self.calculate_gini_from_pd(bins)
            _, n_significant = self.binomial_test_bins(bins)
            reversal_results = self.check_reversals(bins)
            homogeneity_results = self.homogeneity_test(bins)
            proportions = pd.Series(bins).value_counts(normalize=True)
            sorted_proportions = (
                proportions.sort_index()
            )  # Sort proportions by bin index
            bell_curve_fail = self.check_bell_curve(sorted_proportions)

            return {
                "simulation_id": sim_id,
                "thresholds": ",".join([f"{t:.1f}" for t in thresholds]),
                "implied_pd": implied_pd,
                "auc": gini_results["auc"],
                "ar": gini_results["ar"],
                "n_binomial_fail": n_significant,
                "n_reversal_fail": reversal_results["n_reversals"],
                "n_homogeneity_fail": homogeneity_results["n_homogeneity_fails"],
                "max_concentration": proportions.iloc[0],
                "min_concentration": (
                    proportions.iloc[-1] if len(
                        proportions) == self.n_ratings else 0
                ),
                "bell_curve_fail": bell_curve_fail,
            }

        if self.verbose:
            print(f"Starting Uniform threshold simulation")
            print(f"  Score range: ({min_score:.2f}, {max_score:.2f})")
            print(f"  Total iterations: {n_simulations}")
            print(
                f"  Parallel jobs: {n_jobs if n_jobs != -1 else 'all cores'}")
            print("-" * 50)

        if self.verbose:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_single)(sim_id, thresholds)
                for sim_id, thresholds in tqdm(
                    enumerate(all_thresholds),
                    total=n_simulations,
                    desc="Simulating Uniform thresholds",
                )
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_single)(sim_id, thresholds)
                for sim_id, thresholds in enumerate(all_thresholds)
            )

        self.simulation_results = (
            pd.DataFrame(results).sort_values(
                "simulation_id").reset_index(drop=True)
        )
        return self.simulation_results

    def analyze_single_threshold(self, thresholds, alpha=0.05):
        """
        Analyze a single threshold configuration in detail.
        """
        bins = self.create_bins_from_thresholds(thresholds)
        implied_pd = self.calculate_implied_pd(bins)
        gini_results = self.calculate_gini_from_pd(bins)
        binomial_results, n_significant = self.binomial_test_bins(bins, alpha)
        reversal_results = self.check_reversals(bins)
        homogeneity_results = self.homogeneity_test(bins)

        return {
            "thresholds": thresholds,
            "bins": bins,
            "implied_pd": implied_pd,
            "gini_metrics": gini_results,
            "binomial_tests": binomial_results,
            "n_significant_bins": n_significant,
            "reversal_check": reversal_results,
            "homogeneity_check": homogeneity_results,
        }
