import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URDF_DIR = ROOT / "src/isaac_so_arm101/robots/trs_so101/urdf"
ORIGINAL_URDF = URDF_DIR / "so_arm101.urdf"
ACT_V5_URDF = URDF_DIR / "so_arm101_act_v5.urdf"
ROBOT_CFG = ROOT / "src/isaac_so_arm101/robots/trs_so101/so_arm101.py"
PICK_PLACE_ENV_CFG = ROOT / "src/isaac_so_arm101/tasks/pick_place/joint_pos_env_cfg.py"

EXPECTED_LIMITS = {
    "shoulder_lift": (-math.radians(106.0), math.radians(106.0)),
    "wrist_flex": (-math.radians(104.0), math.radians(104.0)),
}

# Global extrema from all 20 episodes in v5 meta/episodes_stats.jsonl.
V5_ACTION_ENVELOPE_DEG = {
    "shoulder_pan": (-65.62637329101562, 6.461538314819336),
    "shoulder_lift": (-104.83516693115234, 51.736263275146484),
    "elbow_flex": (-60.52747344970703, 96.74725341796875),
    "wrist_flex": (33.67033004760742, 102.85713958740234),
    "wrist_roll": (-125.67032623291016, 9.09890079498291),
}

V5_STATE_ENVELOPE_DEG = {
    "shoulder_pan": (-64.83516693115234, 6.02197790145874),
    "shoulder_lift": (-104.0879135131836, 52.659339904785156),
    "elbow_flex": (-55.16483688354492, 96.48351287841797),
    "wrist_flex": (35.25274658203125, 101.62637329101562),
    "wrist_roll": (-125.36264038085938, 8.615385055541992),
}

V5_RESET_ENVELOPE_DEG = {
    "shoulder_pan": (1.1208792, 1.1208792),
    "shoulder_lift": (-103.27911, -103.27911),
    "elbow_flex": (96.08353, 96.08353),
    "wrist_flex": (55.89451, 55.89451),
    "wrist_roll": (-2.3912086, -2.3912086),
}


def _joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        limits[joint.attrib["name"]] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return limits


class ActV5JointLimitsTest(unittest.TestCase):
    def test_dedicated_urdf_changes_only_intended_joint_limits(self):
        original = _joint_limits(ORIGINAL_URDF)
        dedicated = _joint_limits(ACT_V5_URDF)

        self.assertEqual(original.keys(), dedicated.keys())
        for joint_name, original_limits in original.items():
            if joint_name in EXPECTED_LIMITS:
                expected_lower, expected_upper = EXPECTED_LIMITS[joint_name]
                actual_lower, actual_upper = dedicated[joint_name]
                self.assertAlmostEqual(actual_lower, expected_lower, places=6)
                self.assertAlmostEqual(actual_upper, expected_upper, places=6)
            else:
                self.assertEqual(dedicated[joint_name], original_limits)

    def test_v5_dataset_envelopes_are_within_dedicated_limits(self):
        dedicated = _joint_limits(ACT_V5_URDF)

        for envelope in (V5_ACTION_ENVELOPE_DEG, V5_STATE_ENVELOPE_DEG, V5_RESET_ENVELOPE_DEG):
            for joint_name, (minimum_deg, maximum_deg) in envelope.items():
                lower, upper = dedicated[joint_name]
                self.assertGreaterEqual(math.radians(minimum_deg), lower, joint_name)
                self.assertLessEqual(math.radians(maximum_deg), upper, joint_name)

    def test_dedicated_robot_cfg_is_wired_into_factory_zero_environment_chain(self):
        robot_cfg_source = ROBOT_CFG.read_text(encoding="utf-8")
        env_cfg_source = PICK_PLACE_ENV_CFG.read_text(encoding="utf-8")

        self.assertIn("SO_ARM101_ACT_V5_CFG", robot_cfg_source)
        self.assertIn("so_arm101_act_v5.urdf", robot_cfg_source)
        self.assertIn("from isaac_so_arm101.robots import SO_ARM101_ACT_V5_CFG", env_cfg_source)
        self.assertEqual(env_cfg_source.count("self.scene.robot = SO_ARM101_ACT_V5_CFG.replace"), 2)
        self.assertIn(
            "class SoArm101PickPlaceCubeActV37EnvCfg(SoArm101PickPlaceCubeActV36EnvCfg)",
            env_cfg_source,
        )


if __name__ == "__main__":
    unittest.main()
