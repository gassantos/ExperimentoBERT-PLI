# -*- coding: utf-8 -*-
__author__ = 'yshao'

import torch
import logging

from transformers import AutoTokenizer

from formatter.Basic import BasicFormatter
from .bert_feature_tool import example_item_to_feature

logger = logging.getLogger(__name__)


class BertPairTextFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.output_mode = config.get('model', 'output_mode')
        # Limites de parágrafos ao expandir formato q_paras/c_paras (BERT-PLI test)
        try:
            self.max_para_q = config.getint("model", "max_para_q")
        except Exception:
            self.max_para_q = 16
        try:
            self.max_para_c = config.getint("model", "max_para_c")
        except Exception:
            self.max_para_c = 32

    # Separador que não ocorre em GUIDs normais ("query_doc")
    _PARA_SEP = "___"

    @classmethod
    def base_guid(cls, guid: str) -> str:
        """Devolve o GUID original, sem sufixo de índice de parágrafo."""
        return guid.split(cls._PARA_SEP)[0]

    def _expand_para_items(self, temp: dict, mode: str) -> list:
        """
        Expande um item no formato caso-nível (``q_paras``/``c_paras``) em
        todos os pares de parágrafos (qi, ci). Cada par recebe um guid
        ``{original_guid}___{qi}_{ci}`` para permitir agregação posterior.
        """
        items = []
        q_paras = temp["q_paras"][: self.max_para_q]
        c_paras = temp["c_paras"][: self.max_para_c]
        label = temp.get("label", 0)
        for qi, q_p in enumerate(q_paras):
            for ci, c_p in enumerate(c_paras):
                items.append({
                    "guid": f"{temp['guid']}{self._PARA_SEP}{qi}_{ci}",
                    "text_a": q_p,
                    "text_b": c_p,
                    "label": label,
                })
        return items

    def process(self, data, config, mode, *args, **params):
        guids = []
        input_ids = []
        attention_mask = []
        token_type_ids = []
        if mode != 'test':
            labels = []

        for temp in data:
            # Expande formato caso-nível; ou usa item plano diretamente
            if "q_paras" in temp:
                flat_items = self._expand_para_items(temp, mode)
            else:
                flat_items = [temp]

            for item in flat_items:
                res_dict = example_item_to_feature(
                    item, self.max_len, self.tokenizer, self.output_mode,
                    mode=mode,
                    cls_token_at_end=False, pad_on_left=False,
                    cls_token_segment_id=0, pad_token_segment_id=0,
                )
                input_ids.append(res_dict['input_ids'])
                attention_mask.append(res_dict['input_mask'])
                token_type_ids.append(res_dict['segment_ids'])
                guids.append(item['guid'])
                if mode != 'test':
                    labels.append(res_dict['label_id'])

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)
        if mode != 'test':
            labels = torch.LongTensor(labels)
            return {'guid': guids, 'input_ids': input_ids, 'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids, 'label': labels}
        else:
            return {'guid': guids, 'input_ids': input_ids, 'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids}





