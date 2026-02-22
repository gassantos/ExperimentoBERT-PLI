# -*- coding: utf-8 -*-
"""Utilitários de construção de features BERT para pares de texto.

Contém a lógica de tokenização, truncamento, padding e montagem dos
vetores ``input_ids``, ``input_mask`` e ``segment_ids`` usada pelos
formatadores :mod:`formatter.nlp.BertPairTextFormatter` e
:mod:`formatter.nlp.BertDocParaFormatter`.
"""
__author__ = 'yshao'


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def example_item_to_feature(example, max_seq_length,
                                 tokenizer, output_mode, mode='test',
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Converte um exemplo em features numéricas para modelos BERT.

    Tokeniza ``text_a`` (e opcionalmente ``text_b``), aplica truncamento,
    adiciona tokens especiais (``[CLS]``, ``[SEP]``), converte para IDs e
    completa com padding até ``max_seq_length``.

    Args:
        example: dicionário com ``text_a``, ``text_b`` (opcional) e ``label``
            (ignorado em modo ``test``).
        max_seq_length: comprimento máximo (em tokens) incluindo especiais.
        tokenizer: instância de ``transformers.PreTrainedTokenizer``.
        output_mode: ``'classification'`` (label inteiro) ou ``'regression'``
            (label float).
        mode: se ``'test'``, ``label_id`` é omitido do retorno.
        cls_token_at_end: se ``True``, coloca ``[CLS]`` no final.
        pad_on_left: se ``True``, aplica padding à esquerda.
        cls_token: token de início (padrão ``'[CLS]'``).
        sep_token: token separador (padrão ``'[SEP]'``).
        pad_token: ID usado para padding (padrão ``0``).
        sequence_a_segment_id: segment ID para a sequência A (padrão ``0``).
        sequence_b_segment_id: segment ID para a sequência B (padrão ``1``).
        cls_token_segment_id: segment ID do token CLS (padrão ``1``).
        pad_token_segment_id: segment ID dos tokens de padding (padrão ``0``).
        mask_padding_with_zero: se ``True``, padding recebe máscara ``0``.

    Returns:
        Dicionário com:
          - ``input_ids``: lista de inteiros de tamanho ``max_seq_length``.
          - ``input_mask``: máscara de atenção (1 real, 0 padding).
          - ``segment_ids``: IDs de segmento A/B.
          - ``label_id``: rótulo int ou float (ausente em modo ``test``).
    """

    tokens_a = tokenizer.tokenize(example['text_a'])

    tokens_b = None
    if 'text_b' in example:
        tokens_b = tokenizer.tokenize(example['text_b'])
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if mode == 'test':
        res_dict = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        }
        return res_dict

    if output_mode == "classification":
        label_id = int(example['label'])
    else:
        label_id = float(example['label'])

    res_dict = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'label_id': label_id,
    }

    return res_dict


