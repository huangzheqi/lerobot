from pathlib import Path

try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except ModuleNotFoundError:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


PORT = "COM24"
ROBOT_ID = "my_awesome_follower_arm"
CALIBRATION_DIR = Path(
    r"E:\so101_clean_project\cache\huggingface\lerobot\calibration\robots\so101_follower"
)


def main() -> None:
    calibration_file = CALIBRATION_DIR / f"{ROBOT_ID}.json"
    if not calibration_file.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        use_degrees=True,
        cameras={},
    )
    robot = SO101Follower(config)

    try:
        robot.bus.connect()
        robot.bus.disable_torque()
        positions = robot.bus.sync_read("Present_Position")

        print(f"Calibration file: {calibration_file}")
        print("Calibrated joint positions:")
        for name, value in positions.items():
            unit = "%" if name == "gripper" else "deg"
            print(f"{name:16s}: {float(value):9.3f} {unit}")
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
