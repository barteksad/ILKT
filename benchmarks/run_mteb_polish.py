"""Example script for benchmarking all datasets constituting the MTEB Polish leaderboard & average scores.
For a more elaborate evaluation, we refer to https://github.com/rafalposwiata/pl-mteb.
"""

from __future__ import annotations
import sys

import logging

from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData

from mteb import MTEB


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

classification_tasks = [
    "CBD",
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
    "AllegroReviews",
    "PAC",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
]

clustering_tasks = ["EightTagsClustering", "PlscClusteringS2S", "PlscClusteringP2P"]

pair_classification_tasks = ["SICK-E-PL", "PPC", "CDSC-E", "PSC"]

sts_tasks = ["SICK-R-PL", "CDSC-R", "STS22", "STSBenchmarkMultilingualSTS"]

tasks = classification_tasks + clustering_tasks + pair_classification_tasks + sts_tasks
model_id = sys.argv[1].strip()

model = SentenceTransformer(
    model_name_or_path=model_id,
    trust_remote_code=True,
    revision="main",
    model_card_data=SentenceTransformerModelCardData(
        language="pl",
        license="apache-2.0",
        model_name=model_id.split("/")[1],
        model_id=model_id,
    ),
    tokenizer_kwargs={
        "model_max_length": 128,
    },
)

evaluation = MTEB(tasks=tasks, task_langs=["pl"])
evaluation.run(
    # TODO: batch size and max length <- not to run into OOM
    model, output_folder=f"results/pl/{model_id.split('/')[-1]}",# batch_size=32
)
