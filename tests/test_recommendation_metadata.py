"""Ensure finite-test-set runs cannot be pooled with latent-mean runs."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
HAS_WANDB = importlib.util.find_spec("wandb") is not None
HAS_REPLAY_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "matplotlib", "torch", "banditeval")
)

if HAS_PANDAS:
    import pandas as pd
    from plot_wandb_simple_regret_curves import (
        derive_method_label_row,
        validate_recommendation_groups,
    )

if HAS_WANDB:
    from download_wandb_simple_regret import (
        BASE_COLUMNS,
        base_row,
        require_baseline_recommendations_for_recovery,
    )


@unittest.skipUnless(HAS_PANDAS, "W&B plotting requires optional pandas dependency")
class RecommendationPlotGroupingTests(unittest.TestCase):
    def test_finite_recommendations_with_different_index_targets_are_separated(self):
        df = pd.DataFrame([
            {
                "policy_family": "gittins",
                "experiment_variant": "gittins_unit_b32_c0.001_prior_default",
                "method_label": "Custom Gittins",
                "recommendation_rule": "finite_population_posterior_mean",
                "recommendation_std_penalty": 0.0,
                "gittins_index_target": index_target,
            }
            for index_target in (None, "finite_population_mean")
        ])
        df["method_label"] = df.apply(derive_method_label_row, axis=1)
        self.assertEqual(df["method_label"].nunique(), 2)
        self.assertIn("latent index", df["method_label"].iloc[0])
        self.assertIn("finite index", df["method_label"].iloc[1])
        first_labels = df["method_label"].copy()
        df["method_label"] = df.apply(derive_method_label_row, axis=1)
        self.assertEqual(first_labels.tolist(), df["method_label"].tolist())
        validate_recommendation_groups(df, "method_label")
        with self.assertRaisesRegex(ValueError, "index targets"):
            validate_recommendation_groups(df, "experiment_variant")

    def test_identical_variants_with_different_targets_are_separated(self):
        rows = [
            {
                "policy_family": "gittins",
                "experiment_variant": "gittins_unit_b32_c0.001_prior_default",
                "method_label": "Custom Gittins",
                "recommendation_rule": rule,
                "recommendation_std_penalty": 0.0,
            }
            for rule in ("posterior_mean", "finite_population_posterior_mean")
        ]
        df = pd.DataFrame(rows)
        df["method_label"] = df.apply(derive_method_label_row, axis=1)
        self.assertEqual(df["method_label"].nunique(), 2)
        validate_recommendation_groups(df, "method_label")
        with self.assertRaisesRegex(ValueError, "mixes Gittins recommendation"):
            validate_recommendation_groups(df, "experiment_variant")

    def test_repeated_derivation_preserves_separation_by_penalty(self):
        rows = [
            {
                "policy_family": "gittins",
                "experiment_variant": "gittins_unit_b32_c0.001_prior_default",
                "recommendation_rule": "finite_population_posterior_mean_minus_std",
                "recommendation_std_penalty": penalty,
            }
            for penalty in (0.5, 1.0)
        ]
        df = pd.DataFrame(rows)
        df["method_label"] = df.apply(derive_method_label_row, axis=1)
        first_labels = df["method_label"].copy()
        df["method_label"] = df.apply(derive_method_label_row, axis=1)
        self.assertEqual(first_labels.tolist(), df["method_label"].tolist())
        self.assertEqual(df["method_label"].nunique(), 2)
        validate_recommendation_groups(df, "method_label")

    def test_missing_legacy_metadata_matches_explicit_latent_mean(self):
        df = pd.DataFrame([
            {"policy_family": "gittins", "method_label": "Gittins"},
            {"policy_family": "gittins", "method_label": "Gittins",
             "recommendation_rule": "posterior_mean", "recommendation_std_penalty": 0.0,
             "gittins_index_target": "latent_mean"},
        ])
        df["method_label"] = df.apply(derive_method_label_row, axis=1)
        self.assertEqual(df["method_label"].nunique(), 1)
        validate_recommendation_groups(df, "method_label")


@unittest.skipUnless(HAS_WANDB, "W&B download requires optional wandb dependency")
class RecommendationDownloadTests(unittest.TestCase):
    def test_normal_download_preserves_finite_population_metadata(self):
        run = SimpleNamespace(
            id="local-test", name="test_recfinite", state="finished", url="",
            config={
                "experiment_variant": "gittins_unit_b32_c0.001_prior_default",
                "recommendation_rule": "finite_population_posterior_mean",
                "recommendation_std_penalty": 0.0,
                "gittins_index_target": "finite_population_mean",
            },
            summary={},
        )
        row = base_row(run)
        self.assertIn("recommendation_rule", BASE_COLUMNS)
        self.assertIn("gittins_index_target", BASE_COLUMNS)
        self.assertEqual(row["recommendation_rule"], "finite_population_posterior_mean")
        self.assertEqual(row["recommendation_std_penalty"], 0.0)
        self.assertEqual(row["gittins_index_target"], "finite_population_mean")

    def test_legacy_recovery_rejects_new_index_independently_of_recommendation(self):
        with self.assertRaisesRegex(ValueError, "recommendation/index targets"):
            require_baseline_recommendations_for_recovery({
                "gittins_index_target": "finite_population_mean",
                "recommendation_rule": "posterior_mean",
                "recommendation_std_penalty": 0.0,
            })
        require_baseline_recommendations_for_recovery({"gittins_index_target": "latent_mean"})

    def test_legacy_recovery_rejects_new_target_even_without_std_penalty(self):
        for rule, penalty in (
            ("finite_population_posterior_mean", 0.0),
            ("finite_population_posterior_mean_minus_std", 1.0),
            ("posterior_mean_minus_std", 1.0),
        ):
            with self.subTest(rule=rule, penalty=penalty):
                with self.assertRaisesRegex(ValueError, "normal download preserves"):
                    require_baseline_recommendations_for_recovery({
                        "recommendation_rule": rule,
                        "recommendation_std_penalty": penalty,
                    })
        for row in ({}, {"recommendation_rule": "posterior_mean"},
                    {"recommendation_rule": "empirical_mean", "recommendation_std_penalty": 0.0}):
            require_baseline_recommendations_for_recovery(row)


@unittest.skipUnless(HAS_REPLAY_DEPENDENCIES, "Replay requires plotting and simulation dependencies")
class GittinsReplayMetadataTests(unittest.TestCase):
    def test_replay_rejects_historical_or_unknown_index_target(self):
        from plot_arm_eval_rollouts import require_supported_gittins_replay

        for metadata in ({}, {"gittins_index_target": "latent_mean"},
                         {"gittins_index_target": "unknown_target"}):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "Replay historical traces"):
                    require_supported_gittins_replay({"gittins_regret": [0.1], **metadata})
        require_supported_gittins_replay({
            "gittins_regret": [0.1], "gittins_index_target": "finite_population_mean",
        })
        require_supported_gittins_replay({"ucb_regret": [0.1]})


if __name__ == "__main__":
    unittest.main()
