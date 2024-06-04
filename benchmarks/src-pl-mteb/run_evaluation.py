import json
from typing import List
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from models import (
    ModelInfo,
    ModelWrapper,
    RetrievalModelWrapper,
    KeyedVectorsModel,
    TransformerModel,
)
from transformers import HfArgumentParser
from tasks import TaskInfo, tasks, new_tasks, get_main_metric
from dataclasses import dataclass, field
from utils import from_dict

import wandb


@dataclass
class PlMtebArgs:
    models_config: str = field(
        metadata={"help": "Path to models config file."},
        default="configs/sentence_transformers.json",
    )

    def load_model_infos(self) -> List[ModelInfo]:
        with open(self.models_config, "r", encoding="utf-8") as config_file:
            return [
                from_dict(ModelInfo, model_info)
                for model_info in json.load(config_file)
            ]


class PlMtebEvaluator:

    def __init__(self, args: PlMtebArgs):
        self.args = args

    def run(self) -> None:
        for model_info in self.args.load_model_infos():
            print(f"Evaluating model: {model_info.model_name}")
            group, name = None, model_info.model_name
            if name.startswith("ILKT"):
                name = name.split("/")[-1]
            # wandb.init(
            #     project="inter-lingual_knowledge_transferring_in_NLP_embeddings",
            #     name=name,
            #     group=group,
            #     resume="allow",
            # )

            base_model = self._prepare_base_model(model_info)
            for task_info in tasks:
                try:
                    model = self._prepare_task_model(base_model, model_info, task_info)
                    task = self._get_task(task_info)
                    eval_splits = (
                        ["validation"] if task_info.name == "MSMARCO-PL" else ["test"]
                    )
                    evaluation = MTEB(tasks=[task], task_langs=["pl"])
                    evaluation.run(
                        model,
                        eval_splits=eval_splits,
                        output_folder=f"results/{name}",
                    )

                except Exception as e:
                    print(f"Error during evaluation of {task_info.name}: {e}")

        wandb.finish()

    @staticmethod
    def _prepare_base_model(model_info: ModelInfo):
        if model_info.model_type == "ST":
            model = SentenceTransformer(model_info.model_name)
            model.eval()
            if model_info.fp16:
                model.half()
        elif model_info.model_type == "T":
            model = TransformerModel(model_info)
        elif model_info.model_type == "SWE":
            model = KeyedVectorsModel(model_info)
        else:
            raise Exception(f"Unknown type of model: {model_info.model_type}.")
        return model

    @staticmethod
    def _prepare_task_model(base_model, model_info: ModelInfo, task_info: TaskInfo):
        if task_info.task_type == "Retrieval":
            return RetrievalModelWrapper(base_model, model_info)
        return ModelWrapper(base_model, model_info)

    @staticmethod
    def _get_task(task_info: TaskInfo):
        task_name = task_info.name
        if task_name in new_tasks:
            return new_tasks.get(task_name)
        return task_name


if __name__ == "__main__":
    parser = HfArgumentParser([PlMtebArgs])
    args = parser.parse_args_into_dataclasses()[0]
    evaluator = PlMtebEvaluator(args)
    evaluator.run()
