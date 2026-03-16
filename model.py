
from pathlib import Path

from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)

class ModelFactory:
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
    
    def _read_chat_template(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"chat_template file not found: {p}")

        if not (template := p.read_text(encoding="utf-8").strip()):
            raise ValueError(f"chat_template file is empty: {p}")
            
        return template
    
    def _supports_assistant_masking(self, template: str) -> bool:
        """
        Check if the chat template is compatible with assistant-only loss masking.

        The {% generation %} tag is required to mark assistant spans, allowing 
        `assistant_only_loss=True` to correctly mask out user/system tokens.
    
        See details:
        https://huggingface.co/docs/trl/v0.28.0/en/sft_trainer 
        """
        return "{% generation %}" in template and "{% endgeneration %}" in template
            
    def create_tokenizer(self, assistant_only_loss=True):
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.init_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = "right"

        custom_chat_template_path = getattr(self.model_cfg, "custom_chat_template", None)
        if custom_chat_template_path:
            tokenizer.chat_template = self._read_chat_template(custom_chat_template_path)
            
        if assistant_only_loss:        
            tmpl = tokenizer.chat_template
            if not tmpl or not self._supports_assistant_masking(tmpl):
                raise ValueError(
                    "`assistant_only_loss=True` requires a chat_template with "
                    "{% generation %} and {% endgeneration %} tags."
                )
                
        return tokenizer

    def create_model(self, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.init_model,
            **kwargs,
            # device_map="auto",        # for distributed training, device_map can't be 'auto'
        )
        
        qc = kwargs.get("quantization_config", None)
        if isinstance(qc, BitsAndBytesConfig) and (
            qc.load_in_4bit or qc.load_in_8bit
        ):
            model = prepare_model_for_kbit_training(model)
            
        return model
