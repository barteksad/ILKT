"""Example script for benchmarking all datasets constituting the MTEB Polish leaderboard & average scores.
For a more elaborate evaluation, we refer to https://github.com/rafalposwiata/pl-mteb.
"""

from __future__ import annotations
import sys

import logging

from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from sentence_transformers.models import Normalize, Pooling, Transformer
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

transformer_model = Transformer(
    model_name_or_path=model_id,
    config_args={"trust_remote_code": True},
    model_args={"trust_remote_code": True},
    tokenizer_args={
        "model_max_length": 512,
    },
)
pooling = Pooling(transformer_model.get_word_embedding_dimension(), "cls")

model = SentenceTransformer(
    modules=[transformer_model, pooling, Normalize()],
    model_card_data=SentenceTransformerModelCardData(
        language="pl",
        license="apache-2.0",
        model_name=model_id.split("/")[1],
        model_id=model_id,
    ),
)


evaluation = MTEB(tasks=tasks, task_langs=["pl"])
evaluation.run(
    model, output_folder=f"results/pl/{model_id.split('/')[-1]}", batch_size=32
)
