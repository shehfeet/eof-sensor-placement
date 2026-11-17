from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
OUTPUT = ROOT / "outputs"
MONTHS = {"march": "2016-03", "june": "2016-06", "september": "2016-09", "december": "2016-12"}
METHODS = ["dopt", "qdeim", "aopt", "uopt", "hybrid", "rmse"]
CANDIDATES = "voxel"
RANK = 9
RIDGE = 0.0001
BUDGET = 16
SENSORS = 21
RANDOM_LAYOUTS = 100
CORES = 60
WORKERS = 2


def configure(core, group, label, month="2016-03"):
    core.DATA_TOPO = str(DATA / "topo.nc")
    core.DATA_3D = str(DATA / f"3D_mean_{month}.nc")
    core.TIME_START, core.TIME_END = (month + "-01", month + "-02")
    core.VALIDATION_DAY, core.TEST_DAY = (month + "-03", month + "-04")
    core.OUT_ROOT = str(OUTPUT / group / ".work")
    core.OUTPUT_PREFIX = label
    core.VALIDATION_MODES = [(RANK, RANK)]
    core.VALIDATION_RIDGES = [RIDGE]
    core.MAX_SENSORS = SENSORS
    Path(core.OUT_ROOT).mkdir(parents=True, exist_ok=True)


def model(group, label):
    return OUTPUT / group / ".work" / f"{label}__frozen_model.npz"
