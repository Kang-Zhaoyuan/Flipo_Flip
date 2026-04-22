TARGET_LENGTH = 0.35
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_OMEGA = 20.0
SETTLE_TIME = 1.5
ENABLE_TELEMETRY = True
ENABLE_PLOTS = True
OUTPUT_DIR_NAME = "outputs"

H_RATIO = 35.0
L_RATIO = 1.4479973796

MATERIALS = {
	1: ("TPU", 1200),
	2: ("PETG", 1270),
	3: ("Nylon", 1140),
	4: ("Aluminum", 2700),
	5: ("Iron/Steel", 7870),
}

CONTACT_SETTINGS = {
	"solimp": "0.9 0.99 0.001",
	"solref": "0.02 1",
	"condim": "4",
	"friction": "1 0.1 0.01",
	"joint_damping": "0.0013",
}
