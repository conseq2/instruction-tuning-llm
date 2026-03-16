from contextlib import contextmanager
from typing import ContextManager

from accelerate.state import PartialState


class Dist:
    def __init__(self, trainer=None):
        self._trainer = trainer
        self._state = PartialState()

    # lazy resolution
    def bind_trainer(self, trainer):
        self._trainer = trainer

    @property
    def trainer(self):
        return self._trainer

    @property
    def accelerator(self):
        return None if self._trainer is None else self._trainer.accelerator

    @property
    def is_main_process(self) -> bool:
        acc = self.accelerator
        return self._state.is_main_process if acc is None else acc.is_main_process

    @property
    def is_local_main_process(self) -> bool:
        acc = self.accelerator
        return self._state.is_local_main_process if acc is None else acc.is_local_main_process

    @property
    def world_size(self) -> int:
        acc = self.accelerator
        return self._state.num_processes if acc is None else acc.num_processes

    @property
    def rank(self) -> int:
        acc = self.accelerator
        return self._state.process_index if acc is None else acc.process_index

    @property
    def local_rank(self) -> int:
        acc = self.accelerator
        return self._state.local_process_index if acc is None else acc.local_process_index

    @property
    def distributed_type(self):
        acc = self.accelerator
        return self._state.distributed_type if acc is None else acc.distributed_type

    def barrier(self) -> None:
        acc = self.accelerator
        if acc is None:
            self._state.wait_for_everyone()
        else:
            acc.wait_for_everyone()

    def main_process_first(self) -> ContextManager:
        acc = self.accelerator
        return self._state.main_process_first() if acc is None else acc.main_process_first()

    def local_main_process_first(self) -> ContextManager:
        acc = self.accelerator
        return self._state.local_main_process_first() if acc is None else acc.local_main_process_first()

    def unwrap_model(self, model):
        acc = self.accelerator
        return model if acc is None else acc.unwrap_model(model)
    
    @contextmanager
    def synchronized(self):
        self.barrier()
        try:
            yield
        finally:
            self.barrier()
