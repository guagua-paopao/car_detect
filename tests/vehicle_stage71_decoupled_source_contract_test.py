from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage71DecoupledSourceContractTests(unittest.TestCase):
    def test_candidate_target_is_opt_in_and_runtime_only(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("option(VCAS_BUILD_STAGE71_CANDIDATE_TEST", cmake)
        self.assertIn(
            "vehicle_stage71_decoupled_real_engine_test.cpp",
            cmake,
        )
        self.assertIn(
            "target_link_libraries(vehicle_stage71_decoupled_real_engine_test PRIVATE\n"
            "        vehicle_tensorrt_adapters)",
            cmake,
        )

    def test_body_and_color_engines_are_initialized_separately(self) -> None:
        source = (
            ROOT / "tests" / "vehicle_stage71_decoupled_real_engine_test.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("body_runner.initialize(body_artifact", source)
        self.assertIn("color_runner.initialize(color_artifact", source)
        self.assertIn("body_results[index].body_type", source)
        self.assertIn("color_results[index].color", source)
        self.assertNotIn("model_registry.v1.json", source)
        self.assertNotIn("vehicle_analytics.yaml", source)

    def test_candidate_hashes_and_input_sizes_are_cli_bound(self) -> None:
        source = (
            ROOT / "tests" / "vehicle_stage71_decoupled_real_engine_test.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("if (argc != 12)", source)
        self.assertIn("argv[2], argv[3], argv[4], argv[5], body_size", source)
        self.assertIn("argv[7], argv[8], argv[9], argv[10], color_size", source)
        self.assertIn("parsed != 224 && parsed != 256", source)


if __name__ == "__main__":
    unittest.main()
