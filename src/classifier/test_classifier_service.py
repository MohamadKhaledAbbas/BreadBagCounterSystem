import sys
import types
import unittest


def _install_test_stubs():
    """
    Provide lightweight stubs for heavy dependencies so the service
    can be imported in the test environment.
    """
    if "cv2" not in sys.modules:
        dummy_cv2 = types.SimpleNamespace(
            cvtColor=lambda arr, flag: arr,
            COLOR_BGR2RGB=0,
            imwrite=lambda *args, **kwargs: True,
        )
        sys.modules["cv2"] = dummy_cv2

    if "imagehash" not in sys.modules:
        sys.modules["imagehash"] = types.SimpleNamespace(phash=lambda image: "hash")

    if "PIL" not in sys.modules:
        pil_module = types.ModuleType("PIL")
        image_module = types.ModuleType("PIL.Image")

        class DummyImage:
            @staticmethod
            def fromarray(arr):
                return arr

        pil_module.Image = DummyImage
        image_module.Image = DummyImage
        sys.modules["PIL"] = pil_module
        sys.modules["PIL.Image"] = image_module


_install_test_stubs()

from src.classifier.BaseClassifier import BaseClassifier
from src.classifier.ClassifierService import ClassifierService


class DummyClassifier(BaseClassifier):
    def __init__(self, responses):
        self.responses = responses
        self.idx = 0

    def load(self, model_path: str):
        return None

    def predict(self, image):
        try:
            response = self.responses[self.idx]
        except IndexError:
            response = self.responses[-1]
        self.idx += 1
        return response


def build_service(responses, **kwargs):
    return ClassifierService(
        DummyClassifier(responses),
        use_voting=True,
        voting_top_k=None,
        **kwargs,
    )


class ClassifierServiceVotingTests(unittest.TestCase):
    def test_weighted_voting_prefers_high_confidence_sum(self):
        responses = [
            ("GY", 0.919),
            ("GY", 0.875),
            ("W", 0.481),
            ("W", 0.472),
            ("W", 0.463),
        ]
        service = build_service(
            responses,
            weighted_score_threshold=0.55,
            weighted_margin_threshold=0.05,
        )
        candidates = [object() for _ in responses]

        roi, label, conf = service._select_best_with_voting(candidates)

        self.assertEqual(label, "GY")
        self.assertIs(roi, candidates[0])
        self.assertAlmostEqual(conf, responses[0][1])

    def test_weighted_voting_marks_unknown_when_uncertain(self):
        responses = [
            ("A", 0.52),
            ("B", 0.50),
            ("A", 0.05),
        ]
        service = build_service(responses)
        candidates = [object() for _ in responses]

        roi, label, conf = service._select_best_with_voting(candidates)

        self.assertEqual(label, "Unknown")
        self.assertIs(roi, candidates[0])
        self.assertAlmostEqual(conf, responses[0][1])


if __name__ == "__main__":
    unittest.main()
