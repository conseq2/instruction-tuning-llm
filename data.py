import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from loguru import logger

import utils

ROLE="role"
CONTENT="content"

SYSTEM="system"
USER="user"
ASSISTANT="assistant"

TOKENIZER_SAFETY_MARGIN=10

class DataPipeline:
    def __init__(self, data_cfg ,tokenizer, max_seq_len, packing):
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.packing = packing
        
        if packing and getattr(self.data_cfg, "truncate", False):
            logger.warning(
                "You are using packing, so we skip data pipeline truncation. "
                "Long samples will be handled by trainer packing."
            )
            self.data_cfg.truncate=False
        
    def _validate(self, messages: List[Dict[str, str]]) -> bool:
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        
        if messages[0].get(ROLE) not in [SYSTEM, USER]:
            return False
        
        if messages[0].get(ROLE) == SYSTEM:
            if not ((messages[0].get(CONTENT) or "").strip()):
                return False
        
        if messages[-1].get(ROLE) != ASSISTANT:
            return False
        
        start_idx = 1 if messages[0].get(ROLE) == SYSTEM else 0
        expected = USER
        for i in range(start_idx, len(messages)):
            msg = messages[i]
            role = msg.get(ROLE)
            content = (msg.get(CONTENT) or "").strip()
            
            if role != expected or not content:
                return False
            
            expected = ASSISTANT if expected == USER else USER
        
        return True
    
    def _get_templated_len(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> int:
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt))
    
    def _truncate(self, msgs: List[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
        msgs = list(msgs)
        while True:
            min_cnt = 3 if msgs[0].get(ROLE) == SYSTEM else 2
            if len(msgs) < min_cnt:
                return None
            
            prompt_len = self._get_templated_len(msgs[:-1], add_generation_prompt=True)
            available_len = self.max_seq_len - prompt_len - TOKENIZER_SAFETY_MARGIN
            
            if available_len > 0:
                msgs[-1] = dict(msgs[-1])
                assistant_content = msgs[-1][CONTENT]
                assistant_tokens = self.tokenizer.encode(assistant_content, add_special_tokens=False)
                msgs[-1][CONTENT] = self.tokenizer.decode(assistant_tokens[:available_len], skip_special_tokens=True).strip()
                
                if msgs[-1][CONTENT]:
                    return msgs
                
            msgs = msgs[:-2]
    
    def _preprocess(self, examples: Dict[str, List[List[Dict[str, str]]]]) -> Dict[str, List[List[Dict[str, str]]]]:
        processed_msgs = []
        
        for msgs in examples["messages"]:
            if not msgs or not self._validate(msgs):
                continue
            
            if self.packing or self._get_templated_len(msgs) <= self.max_seq_len:
                processed_msgs.append(msgs)
                continue
            
            if not getattr(self.data_cfg, "truncate", False):
                continue
            
            truncated_msgs = self._truncate(msgs)
            if truncated_msgs:
                processed_msgs.append(truncated_msgs)
            
        return {"messages": processed_msgs}

    def _convert_to_messages_dict_jsonl(self) -> str:
        p = Path(self.data_cfg.data_path)
        
        with p.open('r', encoding="utf-8") as f:
            if isinstance(first := json.loads(next(line for line in f if line.strip())), dict):
                if "messages" not in first:
                    raise ValueError(
                        "Invalid jsonl format. dict input must contain 'messages'. "
                        "See sample_dialogue for the expected format."
                    )
                return str(p)
        
        if not isinstance(first, list):
            raise ValueError(
                "Invalid jsonl format. Each line must be either a list of conversations or a dict with a 'messages' field. "
                "See sample_dialogue for the expected format."
            )
            
        fd, temp_p = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)

        with p.open('r', encoding="utf-8") as fi, open(temp_p, 'w', encoding="utf-8") as fo:
            for l in fi:
                if l.strip():
                    x = json.loads(l)
                    if not isinstance(x, list):
                        raise ValueError(
                            "Invalid jsonl format. Each line must have the same type. "
                            "See sample_dialogue for the expected format."
                        )
                    fo.write(json.dumps({"messages": x}, ensure_ascii=False) + "\n")

        return temp_p
    
    def _log_dataset_summary(
        self, 
        *, 
        num_total, 
        num_after, 
        num_train, 
        num_valid
    ):
        msg = utils.build_dataset_log(
            data_cfg=self.data_cfg,
            num_total=num_total,
            num_after=num_after,
            num_train=num_train,
            num_valid=num_valid,
        )
        logger.info("\n" + msg)
    
    def build(self):
        path = self.data_cfg.data_path
        try:
            path = self._convert_to_messages_dict_jsonl()
            ds = load_dataset("json", data_files=path, split="train")
        finally:
            if Path(path).resolve() != Path(self.data_cfg.data_path).resolve() and os.path.exists(path):
                os.remove(path)
    
        num_total = len(ds)
        
        cpu_count = os.cpu_count() or 1
        num_proc = getattr(self.data_cfg, "num_proc", max(1, cpu_count // 4))
        
        ds = ds.map(
            self._preprocess,
            batched=True,
            num_proc=num_proc,
            remove_columns=ds.column_names,
            desc="Data preprocessing",
            load_from_cache_file=True
        )
        num_after = len(ds)

        valid_ratio = float(getattr(self.data_cfg, "valid_ratio", 0.0))
        seed = getattr(self.data_cfg, "seed", 42)
        shuffle = getattr(self.data_cfg, "shuffle", True)

        if not (0.0 <= valid_ratio < 1.0):
            raise ValueError(f"valid_ratio must be in [0, 1), got {valid_ratio}")

        if valid_ratio > 0.0:
            split = ds.train_test_split(
                test_size=valid_ratio, 
                seed=seed, 
                shuffle=shuffle,
                load_from_cache_file=True
            )
            train_ds, valid_ds = split["train"], split["test"]
        else:
            train_ds, valid_ds = ds, None
        
        num_train = len(train_ds)
        num_valid = len(valid_ds) if valid_ds is not None else 0
        
        self._log_dataset_summary(
            num_total=num_total,
            num_after=num_after,
            num_train=num_train,
            num_valid=num_valid,
        )
        
        return train_ds, valid_ds
