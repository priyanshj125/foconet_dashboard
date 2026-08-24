import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = './Train/'
TRAIN_DATA = TRAIN_DIR + 'Train_set_g.npy'

TEST_DIR = './Test/'
TEST_DATA = TEST_DIR + 'Test_set.npy'

DEV_DIR = BASE_DIR + '/'
DEV_DATA = os.path.join(BASE_DIR, 'XQ_dataset_62.npy')

MODEL_PATH = os.path.join(BASE_DIR, 'FocoNet_Full.pth')
CATALOG_PATH = os.path.join(BASE_DIR, 'catalog_ISC_summary.csv')
INVENTORY_PATH = os.path.join(BASE_DIR, 'XQ_inventory.xml')
FAULTS_PATH = os.path.join(BASE_DIR, 'gem_faults.geojson')
PREDICTIONS_CACHE = os.path.join(BASE_DIR, 'predictions_xq62.csv')

MAX_STATIONS = 32
XY_SCALE = 50.0
Z_SCALE = 20.0

BATCH_SIZE = 1024
LR = 0.0001
NUM_EPOCHS = 400

# 11-column station feature vector (concatenated with xyz → 14-D token)
FEATURE_NAMES = [
    'Polarity',
    'ln(maxS/maxP)',
    'ln(Sr/Pr)',
    'ln(Sr/Pz)',
    'ln(St/Pr)',
    'ln(St/Pz)',
    'ln(Sz/Pr)',
    'ln(Sz/Pz)',
    'ln(SNR_R)',
    'ln(SNR_T)',
    'ln(SNR_Z)',
]

FEATURE_HELP = [
    'P-wave first-motion polarity (+1 compression, −1 dilatation)',
    'Broadband ratio ln(max(Sr, St, Sz) / max(Pr, Pz))',
    'Natural-log radial S over radial P',
    'Natural-log radial S over vertical P',
    'Natural-log transverse S over radial P',
    'Natural-log transverse S over vertical P',
    'Natural-log vertical S over radial P',
    'Natural-log vertical S over vertical P',
    'Natural-log radial-component signal-to-noise ratio',
    'Natural-log transverse-component signal-to-noise ratio',
    'Natural-log vertical-component signal-to-noise ratio',
]

LOCATION_NAMES = ['Δx (km)', 'Δy (km)', 'Δz (km)']
LOCATION_HELP = [
    'East-west offset from hypocenter (model input is ÷50)',
    'North-south offset from hypocenter (model input is ÷50)',
    'Vertical offset from hypocenter (model input is ÷20)',
]

# Slider ranges for interactive editing (min, max, step)
FEATURE_SLIDER_RANGES = [
    (-1.0, 1.0, 2.0),   # polarity (±1)
    (-3.0, 5.0, 0.05),  # ln(maxS/maxP)
    (-3.0, 5.0, 0.05),
    (-3.0, 5.0, 0.05),
    (-3.0, 5.0, 0.05),
    (-3.0, 5.0, 0.05),
    (-3.0, 5.0, 0.05),
    (-3.0, 5.0, 0.05),
    (0.0, 10.0, 0.05),  # ln SNR
    (0.0, 10.0, 0.05),
    (0.0, 10.0, 0.05),
]
LOCATION_SLIDER_RANGES = [
    (-120.0, 120.0, 0.5),  # Δx
    (-120.0, 120.0, 0.5),  # Δy
    (0.0, 40.0, 0.5),      # Δz
]

DEMO_EVENT_ID = 'DEMO_1STA_M3.5'
