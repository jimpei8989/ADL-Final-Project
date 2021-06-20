from dataclasses import dataclass, field
from typing import List, Dict

from utils.io import json_load


@dataclass
class Slot:
    name: str
    description: str
    is_categorical: bool
    possible_values: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.is_categorical:
            assert len(self.possible_values) > 0

    @property
    def num_possible_values(self):
        return len(self.possible_values)


@dataclass
class Service:
    service_name: str
    description: str = "No description"
    slots: List[Slot] = field(default_factory=list)
    intents: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.slots = [Slot(**s) for s in self.slots]
        self.slot_by_name = {s.name: s for s in self.slots}

    @property
    def name(self):
        return self.service_name


@dataclass
class Schema:
    services: List[Service] = field(default_factory=list)

    def __post_init__(self):
        self.services = [Service(**s) for s in self.services]
        self.service_by_name = {s.name: s for s in self.services}

    def __iter__(self):
        for s in self.services:
            yield(s)

    @classmethod
    def load_json(cls, json_path):
        return cls(services=json_load(json_path))

    def get_service(self, name: str) -> Service:
        return self.service_by_name[name]
