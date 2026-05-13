from __future__ import annotations
import inspect
import json
from pathlib import Path
from abc import ABC, abstractmethod

import sys
sys.path.append('')

from core.attributed_graph import AttributedGraph

class BaseGenerator(ABC):
    """
    Classe base para todos os geradores de grafos da biblioteca

    Qualquer gerador concreto deve herdar esta classe e implementar o método `generate`, retornando um AttributedGraph
    """

    def __init__(self, name:str, supports_mimic:bool = False, supports_augment:bool = False):
        """
        Parâmetros
        ----------

        name: str
            Nome do gerador

        supports_mimic: bool
            Variável responsável por verificar se o algoritmo faz mimicagem

        supports_augment: bool
            Variável responsável por verificar se o algoritmo faz data augmentation
        """
        self.name = name
        self.supports_mimic = supports_mimic
        self.supports_augment = supports_augment

    @abstractmethod
    def generate(self) -> AttributedGraph:
        """
        Gera um grafo com atributos e devolve um AttributedGraph.

        Este método é abstrato: cada gerador concreto deve implementar a sua própria
        lógica de geração de grafo, features x e y
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"supports_mimic={self.supports_mimic}, "
            f"supports_augment={self.supports_augment}"
            f")"
        )
    
    def mimic(self, base_graph: AttributedGraph, **kwargs) -> AttributedGraph:
        raise NotImplementedError(
            f"{self.__class__.__name__} não suporta mimic(). "
            "Verifique o atributo supports_mimic."
        )
    
    @classmethod
    def from_config(cls, config: dict):
        sig = inspect.signature(cls.__init__)
        allowed_keys = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k,v in config.items() if k in allowed_keys}
        return cls(**filtered)
    
    @classmethod
    def from_json(cls, source):
        if isinstance(source, (str, Path)):
            with open(source, "r") as f:
                config = json.load(f)
        elif isinstance(source, dict):
            config = source
        else:
            raise TypeError("from_json aceita str/Path ou dict")

        # Se você quiser suportar o formato {"model": "...", "params": {...}}
        if "params" in config:
            config = config["params"]

        return cls.from_config(config)

    @classmethod
    def fom_json(cls, source):
        return cls.from_json(source)

    def to_config(self) -> dict:
        # Pega os atributos públicos do objeto (ou um subconjunto que você definir)
        attrs = self.__dict__.copy()
        # Remove coisas internas, se houver
        for key in ["name", "supports_mimic", "supports_augment"]:
            attrs.pop(key, None)
        return attrs
