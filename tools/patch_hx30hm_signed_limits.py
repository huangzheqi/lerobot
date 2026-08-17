from pathlib import Path
import shutil


TABLES_PATH = Path(
    r"E:\so101_clean_project\lerobot\src\lerobot\motors\feetech\tables.py"
)


def main() -> None:
    if not TABLES_PATH.is_file():
        raise FileNotFoundError(f"LeRobot table file not found: {TABLES_PATH}")

    data = TABLES_PATH.read_bytes()
    min_entry = b'    "Min_Position_Limit": 15,'
    max_entry = b'    "Max_Position_Limit": 15,'

    if min_entry in data and max_entry in data:
        print(f"Signed position limits are already enabled in: {TABLES_PATH}")
        return

    newline = b"\r\n" if b"\r\n" in data else b"\n"
    marker = b'    "Homing_Offset": 11,' + newline
    if marker not in data:
        raise RuntimeError(
            "Could not find STS_SMS_SERIES_ENCODINGS_TABLE/Homing_Offset marker. "
            "The installed LeRobot version needs a manual compatibility check."
        )

    replacement = marker + min_entry + newline + max_entry + newline
    patched = data.replace(marker, replacement, 1)

    backup_path = TABLES_PATH.with_suffix(".py.bak")
    if not backup_path.exists():
        shutil.copy2(TABLES_PATH, backup_path)

    TABLES_PATH.write_bytes(patched)
    print(f"Patched: {TABLES_PATH}")
    print(f"Backup:  {backup_path}")
    print("Added sign-magnitude encoding for Min/Max_Position_Limit.")


if __name__ == "__main__":
    main()
