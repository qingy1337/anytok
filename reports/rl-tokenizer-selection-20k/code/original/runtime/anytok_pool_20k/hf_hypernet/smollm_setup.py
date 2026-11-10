"""Prepare a tied-weight causal LM for input-only tokenizer experiments."""

import copy


def separate_input_embeddings_and_freeze_output(model):
    """Copy the input table, preserve the pretrained head, and freeze both tables.

    The copied input table is a fixed reference for the hypernetwork warmup and
    for canonical continuation tokens. The transformer backbone remains
    trainable. Calling this on an already separated model is an error because
    it would silently replace a potentially trained input table.
    """
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("The model must have input embeddings and an output head")
    if input_embeddings.weight is not output_embeddings.weight:
        raise ValueError("Expected tied input and output weights")

    model.config.tie_word_embeddings = False
    model.set_input_embeddings(copy.deepcopy(input_embeddings))
    model.get_input_embeddings().requires_grad_(False)
    model.get_output_embeddings().requires_grad_(False)
    return model
