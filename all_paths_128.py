from pathlib import Path

ROOT="your_root_path"

paths = {
        "tmars_top0to10p": Path("${ROOT}/tmars_bucket/tmars_top0to10p/"),
        "tmars_top10to20p": Path("${ROOT}/tmars_bucket/tmars_top10to20p"),
        "tmars_top20to30p": Path("${ROOT}/tmars_bucket/tmars_top20to30p"),
        "tmars_top30to40p": Path("${ROOT}/tmars_bucket/tmars_top30to40p"),
        "tmars_top0to40p_random25p": Path("${ROOT}/tmars_bucket/tmars_0%_to_40%_random25%"),
        }

alt_name = {
    "tmars_top0to10p": "Top 10%",
    "tmars_top10to20p": "Top 10%-20%",
    "tmars_top20to30p": "Top 20%-30%",
    "tmars_top30to40p": "Top 30%-40%",
    "tmars_top0to40p_random25p": "Top 0%-40% Random 25% Data",

}

samples_per_epoch_dict = {
    "tmars_top0to10p": 12_800_000,
    "tmars_top10to20p": 12_800_000,
    "tmars_top20to30p": 12_800_000,
    "tmars_top30to40p": 12_800_000,
    "tmars_top0to40p_random25p": 12_800_000,
}

match_with_dict = {
    "tmars_top0to10p": "epoch",
    "tmars_top10to20p": "epoch",
    "tmars_top20to30p": "epoch",
    "tmars_top30to40p": "epoch",
    "tmars_top0to40p_random25p": "epoch",
}

subsample_every_dict = {
    "tmars_top0to10p": 1,
    "tmars_top10to20p": 1,
    "tmars_top20to30p": 1,
    "tmars_top30to40p": 1,
    "tmars_top0to40p_random25p": 1,
}