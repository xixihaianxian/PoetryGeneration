common_punctuations = [
    "、",
    "，",
    "。",
    "？",
    "！",
    "：",
    "；",
    "「",
    "」",
    "《",
    "》",
    "（",
    "）",
    "　"
]

mask={"PAD":0,"UNK":1}

MAX_LENGTH=30
SEQ_LENGTH=10
HIDDEN_SIZE = 256
EMBEDDING_DIM = 128
NUM_LAYERS=2
TEMPERATURE=0.8
GENERATION_MAX_LENGTH=40