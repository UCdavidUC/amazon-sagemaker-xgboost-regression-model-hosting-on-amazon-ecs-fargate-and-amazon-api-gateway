import unittest

from backend.common.errors import InvalidRequestError, ModelLoadError, UnknownRouteError
from backend.common.models import get_model, reset_cache
from backend.common.models.features import parse_features


VALID_FEATURES = [0.12, -0.45, 0.78, -0.23, 0.56, -0.89, 1.23, -0.67]


class FeatureParsingTests(unittest.TestCase):
    def test_parse_features_array(self):
        self.assertEqual(parse_features({"features": VALID_FEATURES}), VALID_FEATURES)

    def test_parse_features_csv(self):
        csv = ",".join(str(v) for v in VALID_FEATURES)
        self.assertEqual(parse_features({"pred_x_csv": csv}), VALID_FEATURES)

    def test_wrong_count_rejected(self):
        with self.assertRaises(InvalidRequestError):
            parse_features({"features": [1, 2, 3]})

    def test_non_numeric_rejected(self):
        bad = VALID_FEATURES[:-1] + ["oops"]
        with self.assertRaises(InvalidRequestError):
            parse_features({"features": bad})

    def test_missing_rejected(self):
        with self.assertRaises(InvalidRequestError):
            parse_features({})


class WeightedModelTests(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_default_weights_prediction(self):
        model = get_model("weighted")
        out = model.run({"features": VALID_FEATURES})
        self.assertEqual(out["model"], "weighted")
        self.assertEqual(out["weight_source"], "default-weights")
        self.assertIsInstance(out["prediction"], float)

    def test_request_weights_used(self):
        model = get_model("weighted")
        weights = [1, 0, 0, 0, 0, 0, 0, 0]
        out = model.run({"features": VALID_FEATURES, "weights": weights})
        self.assertEqual(out["weight_source"], "request-weights")
        # dot([f], [1,0,...]) == first feature
        self.assertAlmostEqual(out["prediction"], VALID_FEATURES[0])

    def test_bad_weights_rejected(self):
        model = get_model("weighted")
        with self.assertRaises(InvalidRequestError):
            model.validate_input({"features": VALID_FEATURES, "weights": [1, 2]})


class TimeSeriesModelTests(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_arima_naive_forecast(self):
        model = get_model("arima")
        out = model.run({"steps": 3})
        self.assertEqual(out["steps"], 3)
        self.assertEqual(len(out["forecast"]), 3)
        self.assertEqual(out["source"], "naive-fallback")

    def test_arima_rejects_exog(self):
        model = get_model("arima")
        with self.assertRaises(InvalidRequestError):
            model.validate_input({"steps": 2, "exog": [[1], [2]]})

    def test_sarima_accepts_exog(self):
        model = get_model("sarima")
        norm = model.validate_input({"steps": 2, "exog": [[1], [2]]})
        self.assertEqual(norm["steps"], 2)

    def test_steps_required(self):
        model = get_model("sarima")
        with self.assertRaises(InvalidRequestError):
            model.validate_input({})

    def test_steps_range_enforced(self):
        model = get_model("arima")
        with self.assertRaises(InvalidRequestError):
            model.validate_input({"steps": 0})
        with self.assertRaises(InvalidRequestError):
            model.validate_input({"steps": 10000})


class XGBoostModelTests(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_missing_artifact_raises(self):
        model = get_model("xgboost")
        with self.assertRaises(ModelLoadError):
            model.run({"features": VALID_FEATURES})

    def test_validation_ok_without_artifact(self):
        model = get_model("xgboost")
        norm = model.validate_input({"features": VALID_FEATURES})
        self.assertEqual(norm["features"], VALID_FEATURES)

    def test_artifact_discovery_prefers_portable_format(self):
        import os
        import tempfile

        from backend.common.config import Config
        from backend.common.models.xgboost_model import XGBoostModel

        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            object.__setattr__(cfg, "model_dir", d)  # Config is frozen
            model = XGBoostModel(config=cfg)

            self.assertEqual(model._discover_artifact(), (None, None))

            open(os.path.join(d, "xgboost-model.pkl"), "w").close()
            self.assertEqual(model._discover_artifact()[0], "pickle")

            open(os.path.join(d, "xgboost-model.json"), "w").close()
            self.assertEqual(model._discover_artifact()[0], "native")

            ubj = os.path.join(d, "xgboost-model.ubj")
            open(ubj, "w").close()
            kind, path = model._discover_artifact()
            self.assertEqual(kind, "native")
            self.assertTrue(path.endswith("xgboost-model.ubj"))


class RegistryTests(unittest.TestCase):
    def test_unknown_model(self):
        with self.assertRaises(UnknownRouteError):
            get_model("does-not-exist")


if __name__ == "__main__":
    unittest.main()
