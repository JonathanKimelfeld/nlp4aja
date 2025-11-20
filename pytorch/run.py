import argparse
import copy
import csv
import datetime
import json
import logging
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from utils import split_train_val
from char_cnn_tagger import CharCNNTagger
from char_lstm_tagger import CharLSTMTagger
from lstm_tagger import LSTMTagger
from mtl_wrapper import MTLWrapper
from data_classes import write_sentences_to_excel
try:
    import load_data
    import split_train_test_val
    import train_test_split
except:
    pass

from debug_run import *
from mask_utils import *

DEBUG_MASK = False

UNKNOWN = 'UNKNOWN'
FLAT = "flat"
MTL = "multitask"
HIERARCHICAL = "hierarchical"
THRESHOLD = 2

def prepare_char_sequence(word, to_ix):
    idxs = []
    for i in range(len(word) - 1):
        curr_char = word[i]
        next_char = word[i+1]
        if curr_char == "'":  # treat letters followed by ' as single character
            continue
        if next_char == "'":
            char = curr_char+next_char
        else:
            char = curr_char
        idxs.append(get_index(char, to_ix))
    if word[-1] != "'":
        idxs.append(get_index(word[-1], to_ix))
    return idxs


def prepare_sequence_for_chars(seq, to_ix, char_to_ix, poses=None, pos_to_ix=None):
    res = []
    if pos_to_ix and poses:
        for (w, _), pos in zip(seq, poses):
            res.append((get_index(w, to_ix),
                        prepare_char_sequence(w, char_to_ix),
                        get_index(pos, pos_to_ix)))
    else:
        for w, _ in seq:
            res.append((get_index(w, to_ix), prepare_char_sequence(w, char_to_ix)))
    return res


def prepare_sequence_for_bpes(seq, to_ix, bpe_to_ix, poses=None, pos_to_ix=None):
    res = []
    if pos_to_ix and poses:
        for (w, bpes), pos in zip(seq, poses):
            res.append((get_index(w, to_ix), [get_index(bpe, bpe_to_ix) for bpe in bpes],
                        get_index(pos, pos_to_ix)))
    else:
        for w, bpes in seq:
            res.append((get_index(w, to_ix),
                        [get_index(bpe, bpe_to_ix) for bpe in bpes]))
    return res


def prepare_sequence_for_words(seq, to_ix, poses=None, pos_to_ix=None):
    idxs = []
    if pos_to_ix and poses:
        for (w, _), pos in zip(seq, poses):
            idxs.append((get_index(w, to_ix),
                        get_index(pos, pos_to_ix)))
    else:
        for w, _ in seq:
            ix = get_index(w, to_ix)
            idxs.append((ix,))
    return idxs


def prepare_target(seq, to_ix, field_idx):
    idxs = []
    for w in seq:
        ix = get_index(w[field_idx], to_ix)
        idxs.append(ix)
    return torch.LongTensor(idxs)


def reverse_dict(to_ix):
    # receives a dictionary with words/tags mapped to indices
    # and returns a dictionary mapping indices to words/tags
    ix_to = {}
    for k, v in to_ix.items():
        ix_to[v] = k
    return ix_to


def get_index(w, to_ix):
    return to_ix.get(w, to_ix[UNKNOWN])


def train(training_data, val_data, model_path, word_dict_path, char_dict_path,
          bpe_dict_path, tag_dict_path, frequencies, word_emb_dim, char_emb_dim,
          hidden_dim, dropout, num_kernels=1000, kernel_width=6, by_char=False,
          by_bpe=False, with_smoothing=False, cnn=False, directions=1, device='cpu',
          save_all_models=False, save_best_model=True, epochs=300, lr=0.1, batch_size=8,
          morph=None, weight_decay=0, loss_weights=(1,1,1,1,1), seed=42, legal_morph=False):

    # training data of shape: [(sent, tags), (sent, tags)]
    # where sent is of shape: [(word, bpe), (word, bpe)], len(sent) == number of words
    # and tags is of shape: [(pos, an1, an2, an3, enc),...], len(tags) == len(sent) == number of words

    field_names = ["pos", "an1", "an2", "an3", "enc"]

    model_path_parts = model_path.split(".")
    dict_path_parts = tag_dict_path.split(".")

    pos_training_data = [(sent, [tag_set[0] for tag_set in tags]) for sent, tags in training_data]
    logger.info(f"Number of sentences in training data: {len(pos_training_data)}")

    pos_model_path = model_path_parts[0] + "-pos." + model_path_parts[1]
    pos_dict_path = dict_path_parts[0] + "-pos." + dict_path_parts[1]

    word_to_ix, char_to_ix, bpe_to_ix, pos_to_ix = prepare_dictionaries(pos_training_data, with_smoothing, frequencies)
    torch.save(pos_to_ix, pos_dict_path)
    torch.save(word_to_ix, word_dict_path)
    torch.save(char_to_ix, char_dict_path)
    torch.save(bpe_to_ix, bpe_dict_path)

    val_sents = None

    if by_char:
        if val_data:
            val_sents = [prepare_sequence_for_chars(val_sent[0], word_to_ix,
                                                    char_to_ix)
                         for i, val_sent in enumerate(val_data)]
        train_sents = [prepare_sequence_for_chars(training_sent[0], word_to_ix,
                                                  char_to_ix)
                       for i, training_sent in enumerate(training_data)]
    elif by_bpe:
        if val_data:
            val_sents = [prepare_sequence_for_bpes(val_sent[0], word_to_ix, bpe_to_ix)
                         for i, val_sent in enumerate(val_data)]
        train_sents = [prepare_sequence_for_bpes(training_sent[0], word_to_ix,
                                                 bpe_to_ix)
                       for i, training_sent in enumerate(training_data)]
    else:
        if val_data:
            val_sents = [prepare_sequence_for_words(val_sent[0], word_to_ix)
                         for i, val_sent in enumerate(val_data)]
        train_sents = [prepare_sequence_for_words(training_sent[0], word_to_ix)
                       for i, training_sent in enumerate(training_data)]
    if val_data:
        logger.info(f"Number of sentences in val data: {len(val_sents)}")
    
        val_poses = [[prepare_target(tag_sets, pos_to_ix, field_idx=0).to(device=device)]
                     # inside a list for MTL wrapper purposes
                     for (val_sent, tag_sets) in val_data]
    else:
        val_poses = None
    train_poses = [[prepare_target(tag_sets, pos_to_ix, field_idx=0).to(device=device)]
                   # inside a list for MTL wrapper purposes
                   for (train_sent, tag_sets) in training_data]

    logger.info("Finished preparing POS data:")
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

    if morph == MTL:

        model_path = model_path

        all_train_field_tags, all_val_field_tags, all_field_dicts = prepare_data_for_mtl(field_names, training_data,
                                                                                         val_data, device,
                                                                                         dict_path_parts)

        logger.info(f"Finished preparing MTL data:")
        logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

        mtl_model = train_tag(train_sents=train_sents, val_pos_list=val_poses, val_sents=val_sents,
                              train_tags=all_train_field_tags,
                              val_tags=all_val_field_tags, model_path=model_path, word_to_ix=word_to_ix,
                              char_to_ix=char_to_ix, bpe_to_ix=bpe_to_ix, pos_tag_dictionary=all_field_dicts,
                              word_emb_dim=word_emb_dim, char_emb_dim=char_emb_dim, hidden_dim=hidden_dim,
                              dropout=dropout, num_kernels=num_kernels, kernel_width=kernel_width, by_char=by_char,
                              by_bpe=by_bpe, cnn=cnn, directions=directions, device=device,
                              save_all_models=save_all_models, save_best_model=save_best_model, epochs=epochs, lr=lr,
                              batch_size=batch_size, weight_decay=weight_decay, loss_weights=loss_weights, seed=seed,
                              legal_morph=legal_morph, mask_train=getattr(args, 'mask_train', False),
                              mask_val=getattr(args, 'mask_val', False),
                              treat_gold_illegal_as_underscore=getattr(args, 'treat_gold_illegal_as_underscore', True))
        return mtl_model

    logger.info("Preparing POS training data:")
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

    pos_tag_dictionary = {value: key for key, value in pos_to_ix.items()}

    pos_model = train_tag(
        train_pos_list=train_poses,
        train_sents=train_sents,
        val_pos_list=val_poses,
        val_sents=val_sents,
        train_tags=train_poses,
        field_idx=-1,
        val_tags=val_poses,
        model_path=pos_model_path,
        word_to_ix=word_to_ix,
        char_to_ix=char_to_ix,
        bpe_to_ix=bpe_to_ix,
        tag_to_ix_list=[pos_to_ix],
        pos_tag_dictionary=pos_tag_dictionary,
        word_emb_dim=word_emb_dim,
        char_emb_dim=char_emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_kernels=num_kernels,
        kernel_width=kernel_width,
        by_char=by_char,
        by_bpe=by_bpe,
        cnn=cnn,
        directions=directions,
        device=device,
        save_all_models=save_all_models,
        save_best_model=save_best_model,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        pos_dict_size=0,
        seed=seed,
        legal_morph=legal_morph,
        mask_train=getattr(args, 'mask_train', False),
        mask_val=getattr(args, 'mask_val', False),
        treat_gold_illegal_as_underscore=getattr(args, 'treat_gold_illegal_as_underscore', True)
    )

    if morph == FLAT or morph == HIERARCHICAL:

        pos_dict_size = 0

        if morph == HIERARCHICAL:
            pos_dict_size = len(pos_to_ix)
            if by_bpe or by_char:
                train_sents = [[(idxs[0], idxs[1], tag_idx)
                                for idxs, tag_idx in zip(sent, sent_tags[0])]
                               for sent, sent_tags in zip(train_sents, train_poses)]
                if val_data:
                    val_sents = [[(word_idx, char_idx, tag_idx)
                                  for (word_idx, char_idx), tag_idx in zip(sent, sent_tags[0])]
                                 for sent, sent_tags in zip(val_sents, val_poses)]
            else:
                train_sents = [[(word_idx, tag_idx)
                                for word_idx, tag_idx in zip(sent, sent_tags[0])]
                               for sent, sent_tags in zip(train_sents, train_poses)]
                if val_data:
                    val_sents = [[(word_idx,tag_idx)
                                  for word_idx, tag_idx in zip(sent, sent_tags[0])]
                                 for sent, sent_tags in zip(val_sents, val_poses)]
        field_models = [pos_model]
        ix_to_pos = reverse_dict(pos_to_ix)
        
        if legal_morph:
            collect_and_write_illegal_tag_statistics(training_data, field_names, model_path, logger=logger)
        
        for field_idx, field_name in enumerate(field_names[1:], start=1):

            logger.info(f"Preparing {field_name} training data:")
            logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

            field_training_data = [(sentence, [tag_set[field_idx] for tag_set in tags]) for sentence, tags in
                                   training_data]

            field_tag_to_ix = prepare_tag_dict(field_training_data)
            field_dict_path = dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
            torch.save(field_tag_to_ix, field_dict_path)

            field_model_path = model_path_parts[0] + f"-{field_name}." + model_path_parts[1]
            if val_data:
                val_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=field_idx).to(device=device)]
                                  for (val_sent, tag_sets) in val_data]
            else:
                val_field_tags = None
            train_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=field_idx).to(device=device)]
                                for (train_sent, tag_sets) in training_data]

            logger.info(f"Finished preparing {field_name} data:")
            logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

            ix_to_field_tag = reverse_dict(field_tag_to_ix)

            if legal_morph:
                scan_field_rules_and_check_constraints(
                    field_idx, field_name, val_poses, ix_to_pos, ix_to_field_tag, field_tag_to_ix
                )
            
            # pack POS + tag_to_ix field mappings for train_tag
            pos_and_field_tag_to_ix = [pos_to_ix, field_tag_to_ix]

            field_model = train_tag(
                train_pos_list=train_poses,
                train_sents=train_sents,
                val_pos_list=val_poses,
                val_sents=val_sents,
                train_tags=train_field_tags,
                field_idx=field_idx,
                val_tags=val_field_tags,
                model_path=field_model_path,
                word_to_ix=word_to_ix,
                char_to_ix=char_to_ix,
                bpe_to_ix=bpe_to_ix,
                tag_to_ix_list=pos_and_field_tag_to_ix,
                pos_tag_dictionary=pos_tag_dictionary,
                word_emb_dim=word_emb_dim,
                char_emb_dim=char_emb_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_kernels=num_kernels,
                kernel_width=kernel_width,
                by_char=by_char,
                by_bpe=by_bpe,
                cnn=cnn,
                directions=directions,
                device=device,
                save_all_models=save_all_models,
                save_best_model=save_best_model,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                pos_dict_size=pos_dict_size,
                seed=seed,
                legal_morph=legal_morph,
                mask_train=getattr(args, 'mask_train', False),
                mask_val=getattr(args, 'mask_val', False),
                treat_gold_illegal_as_underscore=getattr(args, 'treat_gold_illegal_as_underscore', True)
            )
            field_models.append(field_model)

        return field_models[0], field_models[1], field_models[2], field_models[3], field_models[4]

    else:
        return pos_model


def train_tag(train_pos_list, train_sents, val_pos_list, val_sents, train_tags, field_idx, val_tags, model_path,
              word_to_ix, char_to_ix,
              bpe_to_ix, tag_to_ix_list, pos_tag_dictionary, word_emb_dim, char_emb_dim,
              hidden_dim, dropout, num_kernels=1000, kernel_width=6, by_char=False,
              by_bpe=False, cnn=False, directions=1, device='cpu',
              save_all_models=False, save_best_model=True, epochs=300, lr=0.1,
              batch_size=8, weight_decay=0, pos_dict_size=0, loss_weights=None, seed=42, legal_morph=False,
              mask_train=False, mask_val=False, treat_gold_illegal_as_underscore=True):
    """
    This is the central function that runs the training process; it trains a model on a given tag (or set of tags),
    with or without early stopping, based on the hyperparameters provided to the function call. It saves and returns
    the best model.
    Trains a single analysis field (field_idx) – or POS when field_idx == -1.
    If `legal_morph` is True and field_idx > -1, predictions are passed
    through `apply_legal_mask` before computing filtered metrics.
    D1 = Raw accuracy (all tokens)
    D2 = Filtered accuracy (tokens with constrained rules)
    D3 = Filtered accuracy with legal-morph masking (tokens with constrained rules and non-null, legal gold)
    """
    if DEBUG_MASK:
        debug_validate_tag_inputs(field_idx, train_pos_list, train_sents, train_tags,
                                  val_pos_list, val_sents, val_tags, tag_to_ix_list)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)


    field_names = ['pos', 'an1', 'an2', 'an3', 'enc']
    field_name = field_names[field_idx] if field_idx >= 0 else 'pos'
    print(f"Training {field_name}")

    pos_tag_to_ix = tag_to_ix_list[0]
    field_tag_to_ix = tag_to_ix_list[1] if len(tag_to_ix_list) > 1 else pos_tag_to_ix

    pos_ix_to_tag = reverse_dict(pos_tag_to_ix)
    field_ix_to_tag = reverse_dict(field_tag_to_ix)

    assert_bijection(field_tag_to_ix, field_ix_to_tag, field_name)

    logits_dim = len(field_ix_to_tag)
    print(f"[vocab] [{field_name}] logits_dim={logits_dim}")

    base_model = base_model_factory(by_char or by_bpe, cnn)
    model = MTLWrapper(
        word_emb_dim, char_emb_dim, hidden_dim, dropout,
        len(word_to_ix),
        len(char_to_ix) if by_char else len(bpe_to_ix),
        [logits_dim],
        num_kernels, kernel_width,
        directions=directions, device=device,
        model_type=base_model, pos_dict_size=pos_dict_size
    )
    
    model = model.to(device=device)

    loss_fn = nn.NLLLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info("Begin training:")
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

    # flatten POS once for masking calls
    train_pos_flat = [p[0] if isinstance(p, (list, tuple)) else p
                      for p in train_pos_list]
    val_pos_flat = [p[0] if isinstance(p, (list, tuple)) else p
                    for p in val_pos_list]

    if legal_morph and field_idx in (1, 2, 3):
        legal_rules_for_field = get_legal_rules_for_field(field_idx)
        sanity_check_mask_activation(
            model=model,
            predict_fn=predict_tags,
            val_sents=val_sents,
            val_pos_flat=val_pos_flat,
            field_tag_to_ix=field_tag_to_ix,
            field_ix_to_tag=field_ix_to_tag,
            pos_ix_to_tag=pos_ix_to_tag,
            field_idx=field_idx,
            field_name=field_name,
            legal_rules_for_field=legal_rules_for_field,
        )

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        # diagnostics per epoch (for masking)
        train_diag = new_diag()

        model.train()
        epoch_loss = 0.0
        token_cnt = 0

        shuffled_idx = random.sample(range(len(train_sents)), len(train_sents))
        for start in range(0, len(shuffled_idx), batch_size):
            batch_ids = shuffled_idx[start:start + batch_size]
            batch_loss = 0.0

            for idx in batch_ids:
                sent = train_sents[idx]
                gold = train_tags[idx]

                model.hidden = model.init_hidden(hidden_dim)
                logits = model(sent)  # list of [Tensors]

                train_diag["total_tokens"] += len(logits[0])

                # Apply train-time masking if enabled (on analysis fields only)
                if legal_morph and mask_train and field_idx > 0:
                    masked_loss = apply_training_mask_and_loss(
                        logits=logits,
                        gold=gold,
                        field_idx=field_idx,
                        field_name=field_name,
                        field_tag_to_ix=field_tag_to_ix,
                        field_ix_to_tag=field_ix_to_tag,
                        pos_ix_to_tag=pos_ix_to_tag,
                        pos_indices_flat=train_pos_flat[idx],
                        train_diag=train_diag,
                        loss_fn=loss_fn,
                        treat_gold_illegal_as_underscore=treat_gold_illegal_as_underscore,
                        legal_values=legal_values,
                    )
                    batch_loss += masked_loss
                else:
                    loss_vec = [loss_fn(l, g) for l, g in zip(logits, gold)]
                    batch_loss += torch.stack(loss_vec).mean()

            batch_loss /= len(batch_ids)
            epoch_loss += batch_loss.item() * len(batch_ids)
            token_cnt += len(batch_ids)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = predict_tags(model, train_sents)
            debug_len_pairs(train_logits, train_tags, ctx="chk/train_raw")
            assert_TV_shapes(train_logits, train_tags, ctx="train_raw")

            train_raw_acc = calculate_accuracy(train_logits, train_tags)

            if legal_morph and mask_val and field_idx > 0:
                train_masked = apply_legal_mask(
                    train_logits, train_pos_flat,
                    field_tag_to_ix, field_ix_to_tag, pos_ix_to_tag,
                    field_idx, field_name
                )
                debug_len_pairs(train_masked, train_tags, ctx="chk/train_filt")
                assert_TV_shapes(train_masked, train_tags, ctx="train_filt")

                normalized_train_logits = normalize_logits_to_list_list(train_logits)

                legal_rules_for_field = get_legal_rules_for_field(field_idx)
                train_filt_acc, train_filt_stats = calculate_accuracy_for_filtered_predictions(
                    train_masked, train_tags,
                    pos_list=train_pos_flat,
                    field_idx=field_idx,
                    field_tag_to_ix=field_tag_to_ix,
                    field_ix_to_tag=field_ix_to_tag,
                    pos_ix_to_tag=pos_ix_to_tag,
                    legal_rules_for_field=legal_rules_for_field,
                    raw_unmasked_scores=normalized_train_logits
                )
            else:
                train_filt_acc = None
                train_filt_stats = None

        if val_sents:
            val_diag = new_diag()

            with torch.no_grad():
                val_logits = predict_tags(model, val_sents)
                debug_len_pairs(val_logits, val_tags, ctx="chk/val_raw")
                assert_TV_shapes(val_logits, val_tags, ctx="val_raw")
                val_raw_acc = calculate_accuracy(val_logits, val_tags)

                val_loss = get_loss_on_val(
                    val_sents, val_tags, val_logits, loss_weights
                )

                if legal_morph and mask_val and field_idx > 0:
                    val_filt_acc, val_filt_stats = apply_validation_mask_metrics(
                        val_logits=val_logits,
                        val_tags=val_tags,
                        val_pos_flat=val_pos_flat,
                        field_idx=field_idx,
                        field_name=field_name,
                        field_tag_to_ix=field_tag_to_ix,
                        field_ix_to_tag=field_ix_to_tag,
                        pos_ix_to_tag=pos_ix_to_tag,
                        logits_dim=logits_dim,
                        val_diag=val_diag,
                        debug_mask=DEBUG_MASK,
                    )
                    select_acc = val_filt_acc
                else:
                    val_filt_acc = None
                    val_filt_stats = None
                    select_acc = val_raw_acc

        train_raw_at_d3_str = f"{train_filt_stats['raw_at_d3']:.4f}" if (train_filt_stats is not None and train_filt_stats.get('raw_at_d3') is not None) else 'n/a'
        val_raw_at_d3_str = f"{val_filt_stats['raw_at_d3']:.4f}" if (val_filt_stats is not None and val_filt_stats.get('raw_at_d3') is not None) else 'n/a'
        logger.info(
            f"Epoch {epoch:3d} | "
            f"train loss {epoch_loss / token_cnt:.4f} | "
            f"train raw (D1) {train_raw_acc:.4f} "
            f"train raw@D3 (D3) {train_raw_at_d3_str} "
            f"train filt (D3) {train_filt_acc if train_filt_acc is not None else 'n/a'} | "
            f"val raw (D1) {val_raw_acc:.4f} "
            f"val raw@D3 (D3) {val_raw_at_d3_str} "
            f"val filt (D3) {val_filt_acc if val_filt_acc is not None else 'n/a'}"
        )

        if legal_morph and field_idx > 0:
            val_diag_for_log = val_diag if (val_sents and 'val_diag' in locals()) else None
            log_mask_diagnostics(train_diag, val_diag_for_log, train_filt_stats, val_filt_stats, field_idx, val_sents)

        if train_diag["masked_tokens"] > 0:
            assert train_diag[
                       "saw_minus_inf"] > 0, "Masking did not write -1e9 into tags during TRAIN (check mask_train path)."

        # early-stopping on filtered (or raw) val loss
        if val_sents and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience == 5:
                logger.info("Early stop: patience exhausted")
                break

        model.train()

    # save best model
    if save_best_model and best_state is not None:
        torch.save(best_state, model_path)

    logger.info("Training finished.")
    return model if best_state is None else best_state


def prepare_data_for_mtl(field_names, training_data, val_data, device, dict_path_parts, test=False):
    all_field_dicts = []
    all_train_field_tags_misordered = []
    all_val_field_tags_misordered = []
    for field_idx, field_name in enumerate(field_names):

        logger.info(f"Preparing {field_name} data:")
        logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

        field_training_data = [(sent, [tag_set[field_idx] for tag_set in tags]) for sent, tags in training_data]

        if not test:
            field_tag_to_ix = prepare_tag_dict(field_training_data)
            field_dict_path = dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
            torch.save(field_tag_to_ix, field_dict_path)
            all_field_dicts.append(field_tag_to_ix)
        else:
            field_dict_path = dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
            field_tag_to_ix = torch.load(field_dict_path)

        if val_data:
            val_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=field_idx).to(device=device)]
                              for (val_sent, tag_sets) in val_data]
        else:
            val_field_tags = None
        train_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=field_idx).to(device=device)]
                            for (train_sent, tag_sets) in training_data]

        all_train_field_tags_misordered.append(train_field_tags)
        all_val_field_tags_misordered.append(val_field_tags)
    all_train_field_tags = reorder_sent_tags(all_train_field_tags_misordered, device)
    if val_data:
        all_val_field_tags = reorder_sent_tags(all_val_field_tags_misordered, device)
    else:
        all_val_field_tags = None

    return all_train_field_tags, all_val_field_tags, all_field_dicts


def reorder_sent_tags(misordered_tags, device):
    """
    This function receives a list of lists of tags of the following shape:
    [[field_tags], [field_tags], [field_tags], [field_tags], [field_tags]] (of length num_fields)
    where each list [field_tags] = [[sent], [sent], [sent]...]
    and each [sent] = [word_tag, word_tag, word_tag...]

    It outputs a list of length num_sentences, where each sentence is:
    sent = [(w1_t1, w2_t1, w3_t1), (w1_t2, w2_t2, w3_t2)....] so len(sent) == num_fields
    :param misordered_tags:
    :return:
    """
    num_fields = len(misordered_tags)

    ordered_sents = []
    for sent_idx, sent in enumerate(misordered_tags[0]):
        new_sent = []
        for field_idx in range(num_fields):
            new_sent.append([misordered_tags[field_idx][sent_idx][0][word_idx] for word_idx in range(len(sent[0]))])
        ordered_sents.append(torch.LongTensor(new_sent).to(device=device))
    return ordered_sents


def prepare_tag_dict(training_data):
    tag_to_ix = {}
    for sent, tags in training_data:
        for word_bpe, tag in zip(sent, tags):
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    if UNKNOWN not in tag_to_ix:
        tag_to_ix[UNKNOWN] = len(tag_to_ix)

    return tag_to_ix


def prepare_dictionaries(training_data, with_smoothing, frequencies):
    word_to_ix = {}
    char_to_ix = {}
    bpe_to_ix = {}
    tag_to_ix = {}

    for sent, tags in training_data:
        for word_bpe, tag in zip(sent, tags):
            word = word_bpe[0]
            bpes = word_bpe[1]
            if word not in word_to_ix:
                if not (with_smoothing and frequencies[word] <= THRESHOLD):
                    word_to_ix[word] = len(word_to_ix)
            for i in range(len(word) - 1):
                curr_char = word[i]
                next_char = word[i + 1]
                if curr_char == "'":
                    continue
                if next_char == "'":  # treat letters followed by ' as single character
                    char = curr_char + next_char
                else:
                    char = curr_char
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
            if word[-1] != "'":
                if word[-1] not in char_to_ix:
                    char_to_ix[word[-1]] = len(char_to_ix)
            if bpes:
                for bpe in bpes:
                    if bpe not in bpe_to_ix:
                        bpe_to_ix[bpe] = len(bpe_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    if UNKNOWN not in char_to_ix:
        char_to_ix[UNKNOWN] = len(char_to_ix)

    if UNKNOWN not in word_to_ix:
        word_to_ix[UNKNOWN] = len(word_to_ix)

    if UNKNOWN not in bpe_to_ix:
        bpe_to_ix[UNKNOWN] = len(bpe_to_ix)

    if UNKNOWN not in tag_to_ix:
        tag_to_ix[UNKNOWN] = len(tag_to_ix)

    return word_to_ix, char_to_ix, bpe_to_ix, tag_to_ix


def base_model_factory(by_char, cnn):
    if not by_char:
        return LSTMTagger
    else:
        if cnn:
            return CharCNNTagger
        else:
            return CharLSTMTagger


def get_loss_on_val(val_sents, val_tags, predicted_tags, loss_weights):
    """
    For giving different tasks different weights (such as giving POS a higher weight than enc)
    :param val_sents:
    :param val_tags:
    :param predicted_tags:
    :param loss_weights:
    :return:
    """
    loss_function = nn.NLLLoss()
    loss = 0.0
    divisor = len(val_sents)
    for tags, predicted in zip(val_tags, predicted_tags):
        if len(predicted) == 0:
            divisor -= 1
            continue
        sent_loss = 0.0
        if not loss_weights:
            loss_weights = [1]*len(predicted)
        for task_pred, task_tags, weight in zip(predicted, tags, loss_weights):
            task_pred = task_pred.view(-1, task_pred.size(-1))
            task_tags = task_tags.view(-1)
            sent_loss += weight*loss_function(task_pred, task_tags)
        avg_sent_loss = sent_loss/sum(loss_weights)
        loss += avg_sent_loss
    return loss/divisor


def predict_tags(model, sents):
    """
    Predicting tags for a given model and a given set of sentences.
    In case the sentence is empty, append an empty list to the predicted tags.
    Upon receiving an invalid sentence, raise an error.
    """
    model = model.eval()
    predicted = []
    invalid_line_counter = 0

    for sent in sents:

        try:
            if len(sent) == 0:
                predicted.append([])
            else:
                predicted_tags = model(sent)
                predicted.append(predicted_tags)

        except RuntimeError as e:
            if 'zero batch' in str(e):
                predicted.append([])
                invalid_line_counter += 1
            else:
                print("An invalid sentence was given. Check input in cleaning stage.")
                raise e

    if invalid_line_counter != 0:
        print(f"{invalid_line_counter} invalid sentences found in the given input.")

    return predicted


def calculate_accuracy(predicted_tag_scores, true_tags_2d):
    """
    Computes accuracy across all sentences and fields using a vectorized approach.
    predicted_tag_scores: list[list[torch.Tensor]] # Sentences -> Fields -> Tensor(num_words, num_tags)
    true_tags_2d: list[list[torch.Tensor]] # Sentences -> Fields -> Tensor(num_words)
    """
    correct, total = 0, 0

    # The vectorized implementation expects a flat list of fields, where prediction
    # tensors are shaped (num_tags, num_words). Raw predictions from the model are
    # (num_words, num_tags) and are nested in sentences, so we flatten and transpose.
    predicted = [field.T for sent in predicted_tag_scores for field in sent]
    gold = [field for sent in true_tags_2d for field in sent]

    for field_scores, gold_field in zip(predicted, gold):
        preds = field_scores.argmax(dim=0)
        correct += (preds.cpu() == gold_field.cpu()).sum().item()
        total += gold_field.numel()

    return (correct / total) if total > 0 else 0.0


def extract_field_logits(predicted_sents, field_idx=0):
    """
    Normalize raw model outputs into a list-of-sentences structure that
    contains a single head per sentence (matching the expected shape for
    masking/metrics helpers).
    """
    extracted = []
    target_idx = 0 if field_idx is None else field_idx
    for sentence_scores in predicted_sents:
        if isinstance(sentence_scores, list):
            if 0 <= target_idx < len(sentence_scores):
                field_scores = sentence_scores[target_idx]
            else:
                field_scores = sentence_scores[0]
        else:
            field_scores = sentence_scores
        field_scores = ensure_2d_tensor(field_scores)
        extracted.append([field_scores])
    return extracted


def select_gold_field_tags(true_tags_nested, field_offset):
    selected = []
    target_idx = max(0, field_offset or 0)
    for sent_fields in true_tags_nested:
        if isinstance(sent_fields, list) and sent_fields:
            idx = target_idx if target_idx < len(sent_fields) else 0
            selected.append([sent_fields[idx]])
        else:
            selected.append(sent_fields)
    return selected


def logits_to_predictions(field_logits):
    """Convert [sentences → [field_tensor]] logits into index predictions."""
    predictions = []
    for sent_fields in field_logits:
        if not sent_fields:
            predictions.append([])
            continue
        field_tensor = ensure_2d_tensor(sent_fields[0])
        if hasattr(field_tensor, "shape") and field_tensor.shape[0] == 0:
            predictions.append([])
            continue
        if hasattr(field_tensor, "dim") and field_tensor.dim() == 2:
            preds = field_tensor.argmax(dim=1).detach().cpu().tolist()
        else:
            tensor_cpu = torch.as_tensor(field_tensor).detach().cpu()
            preds = tensor_cpu.argmax(dim=1).tolist() if tensor_cpu.dim() == 2 else []
        predictions.append(preds)
    return predictions


def combine_field_predictions(per_field_predictions):
    """Transpose list[field][sentence] → list[sentence][field]."""
    if not per_field_predictions:
        return []
    num_sentences = len(per_field_predictions[0])
    combined = []
    for sent_idx in range(num_sentences):
        sentence_fields = []
        for field_preds in per_field_predictions:
            if sent_idx < len(field_preds):
                sentence_fields.append(field_preds[sent_idx])
            else:
                sentence_fields.append([])
        combined.append(sentence_fields)
    return combined


def flatten_pos_sequences(pos_sequences):
    """Normalize POS sequences into torch.LongTensor per sentence."""
    if not pos_sequences:
        return []
    flattened = []
    for entry in pos_sequences:
        if isinstance(entry, (list, tuple)):
            if not entry:
                flattened.append(torch.LongTensor([]))
                continue
            candidate = entry[0]
        else:
            candidate = entry
        if isinstance(candidate, torch.Tensor):
            flattened.append(candidate.detach().cpu())
        else:
            flattened.append(torch.LongTensor(candidate))
    return flattened


def warn_if_filtered_lt_raw(stats, split_label):
    raw_at_d3_val = stats.get('raw_at_d3')
    if raw_at_d3_val is None or stats.get('D3', 0) == 0:
        return
    filtered = stats.get('filtered_accuracy')
    if filtered is None:
        return
    if filtered + 1e-6 < raw_at_d3_val:
        print(f"[warn {split_label}] filtered({filtered:.4f}) < raw@D3({raw_at_d3_val:.4f}) "
              f"on D3={stats['D3']}")


def log_filtered_policy_stats(split_label, field_name, stats):
    raw_at_d3_val = stats.get('raw_at_d3')
    raw_at_d3_str = (f"{raw_at_d3_val:.4f}" if raw_at_d3_val is not None
                     else "n/a" if stats.get('D3', 0) == 0 else "n/a")
    print(f"[{split_label} filtered] Exclude policy over D3 ({field_name}): "
          f"D1={stats['D1']} D2={stats['D2']} D3={stats['D3']} "
          f"raw@D3={raw_at_d3_str} filtered={stats['filtered_accuracy']:.4f} | "
          f"unconstrained={stats['unconstrained_tokens']} "
          f"null_gold={stats['null_gold']} "
          f"gold_illegal={stats['gold_illegal']} "
          f"masked={stats['masked_tokens']}")
    print("[policy] Filtered accuracy computed with Exclude policy over D3 "
          "(constrained & non-null & non-illegal-gold tokens only). "
          "Fair comparison: filtered ≥ raw@D3")


def test(test_data, model_path, word_dict_path, char_dict_path, bpe_dict_path,
         tag_dict_path, word_emb_dim, char_emb_dim, hidden_dim, dropout,
         num_kernels, kernel_width, by_char=False, by_bpe=False, out_path=None,
         cnn=False, directions=1, device='cpu', morph=False, use_true_pos=False,
         test_sent_sources=None, enforce_legal_morphology=False, mask_val=False):
    """

    Prepares all the data, and then calls the function that actually runs testing
    """

    if not out_path:
        out_path = str(datetime.date.today())

    field_names = ["pos", "an1", "an2", "an3", "enc"]

    model_path_parts = model_path.split(".")
    dict_path_parts = tag_dict_path.split(".")

    word_to_ix = torch.load(word_dict_path)
    char_to_ix = torch.load(char_dict_path)
    bpe_to_ix = torch.load(bpe_dict_path)

    test_words = [[word[0] for word in test_sent[0]] for test_sent in test_data]

    if by_char:
        test_sents = [prepare_sequence_for_chars(test_sent[0], word_to_ix, char_to_ix)
                      for test_sent in test_data]
    elif by_bpe:
        test_sents = [prepare_sequence_for_bpes(test_sent[0], word_to_ix, bpe_to_ix)
                      for test_sent in test_data]
    else:
        test_sents = [prepare_sequence_for_words(test_sent[0], word_to_ix)
                      for test_sent in test_data]

    if morph == MTL:

        model_path = model_path

        all_test_field_tags, _, _ = prepare_data_for_mtl(field_names, test_data, None, device, dict_path_parts, test=True)
        all_field_dict_paths = [dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
                                for field_name in field_names]

        logger.info(f"Finished preparing MTL data:")
        logger.info(datetime.datetime.now().strftime("%H:%M:%S"))

        mtl_results = test_morph_tag(test_sents=test_sents, test_field_tags=all_test_field_tags, test_words=test_words,
                                     model_path=model_path, word_dict=word_to_ix, char_dict=char_to_ix,
                                     bpe_dict=bpe_to_ix, tag_dict_path_list=all_field_dict_paths,
                                     word_emb_dim=word_emb_dim, char_emb_dim=char_emb_dim, hidden_dim=hidden_dim,
                                     dropout=dropout, num_kernels=num_kernels, kernel_width=kernel_width,
                                     legal_morph=enforce_legal_morphology, by_char=by_char, by_bpe=by_bpe,
                                     out_path=out_path, cnn=cnn, directions=directions, device=device,
                                     field_names=field_names, test_sent_sources=test_sent_sources,
                                     mask_inference=mask_val)
        return mtl_results

    results = []

    pos_model_path = model_path_parts[0] + f"-pos." + model_path_parts[1]
    pos_dict_path = dict_path_parts[0] + f"-pos." + dict_path_parts[1]
    pos_out_path = out_path + f"-pos"

    pos_tag_to_ix = torch.load(pos_dict_path)
    ix_to_pos = reverse_dict(pos_tag_to_ix)
    test_pos_tags = [[prepare_target(tag_sets, pos_tag_to_ix, field_idx=0).to(device=device)]
                       for (train_sent, tag_sets) in test_data]

    pos_results = test_morph_tag(test_sents=test_sents, test_field_tags=test_pos_tags, test_pos_tags=test_pos_tags, test_words=test_words,
                                 field_index=None, model_path=pos_model_path, word_dict=word_to_ix,
                                 char_dict=char_to_ix,
                                 bpe_dict=bpe_to_ix, tag_dict_path_list=[pos_dict_path], word_emb_dim=word_emb_dim,
                                 char_emb_dim=char_emb_dim, hidden_dim=hidden_dim, dropout=dropout,
                                 num_kernels=num_kernels, kernel_width=kernel_width, legal_morph=False, by_char=by_char,
                                 by_bpe=by_bpe, out_path=pos_out_path, cnn=cnn, directions=directions, device=device,
                                 field_names=["pos"], return_shaped_results=(morph == HIERARCHICAL),
                                 test_sent_sources=test_sent_sources)
    results.append(pos_results)

    if morph == FLAT or morph == HIERARCHICAL:

        pos_dict_size = 0

        if morph == HIERARCHICAL:
            if use_true_pos:
                test_pos = [tags[0] for tags in test_pos_tags]
            else:
                test_pos = [tags[0] if tags else [] for tags in pos_results[1]]
            if by_bpe or by_char:
                test_sents = [[(idxs[0], idxs[1], tag_idx) for idxs, tag_idx in zip(sent, sent_tags)]
                              for sent, sent_tags in zip(test_sents, test_pos)]
            else:
                test_sents = [[(word_idx, tag_idx) for word_idx, tag_idx in zip(sent, sent_tags)]
                              for sent, sent_tags in zip(test_sents, test_pos)]
            pos_dict_size = len(pos_tag_to_ix)
            results[0] = pos_results[0][0]
        else:
            results[0] = pos_results[0]

        for field_idx, field_name in enumerate(field_names[1:], start=1):
            field_idx += 1

            field_model_path = model_path_parts[0] + f"-{field_name}." + model_path_parts[1]
            field_dict_path = dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
            field_out_path = out_path + f"-{field_name}"

            field_tag_to_ix = torch.load(field_dict_path)
            test_field_tags = [[prepare_target(tag_sets, field_tag_to_ix, field_idx=field_idx).to(device=device)]
                               for (train_sent, tag_sets) in test_data]

            field_results = test_morph_tag(test_sents, test_field_tags, test_pos_tags, test_words,
                                           field_idx,
                                           field_model_path, word_to_ix, char_to_ix, bpe_to_ix, [field_dict_path],
                                           word_emb_dim, char_emb_dim, hidden_dim, dropout, num_kernels, kernel_width,
                                           enforce_legal_morphology, by_char, by_bpe, field_out_path, cnn, directions,
                                           device,
                                           pos_dict_size, field_names=[field_name],
                                           test_sent_sources=test_sent_sources,
                                           mask_inference=mask_val,
                                           pos_ix_to_tag=ix_to_pos)
            results.append(field_results[0])

        return results

    else:
        return pos_results[0]


def test_morph_tag(test_sents, test_field_tags, test_pos_tags, test_words, field_index, model_path,
                   word_dict, char_dict, bpe_dict, tag_dict_path_list, word_emb_dim, char_emb_dim, hidden_dim,
                   dropout, num_kernels, kernel_width, legal_morph, by_char=False, by_bpe=False, out_path=None,
                   cnn=False, directions=1, device='cpu', pos_dict_size=0, return_shaped_results=False,
                   field_names=None, test_sent_sources=None, mask_inference=False, pos_ix_to_tag=None):
    if not out_path:
        out_path = str(datetime.date.today())

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    tag_dict_list = [torch.load(tag_dict_path) for tag_dict_path in tag_dict_path_list]

    ix_to_tag_list = [reverse_dict(tag_dict) for tag_dict in tag_dict_list]
    base_model = base_model_factory(by_char or by_bpe, cnn)

    model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout, len(word_dict),
                       len(char_dict) if by_char else len(bpe_dict), [len(tag_dict) for tag_dict in tag_dict_list],
                       num_kernels, kernel_width, directions=directions, device=device, pos_dict_size=pos_dict_size,
                       model_type=base_model)

    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model = model.to(device=device)

    tag_scores = predict_tags(model, test_sents)

    num_fields = len(tag_dict_list)
    if not field_names or len(field_names) != num_fields:
        if not field_names:
            field_names = [f"{i}" for i in range(num_fields)]
        elif len(field_names) != num_fields:
            field_names = [f"{i}" for i in range(num_fields)]

    pos_list = flatten_pos_sequences(test_pos_tags)
    all_field_logits = [extract_field_logits(tag_scores, idx) for idx in range(num_fields)]
    gold_per_field = [select_gold_field_tags(test_field_tags, idx) for idx in range(num_fields)]

    effective_pos_ix = pos_ix_to_tag
    if effective_pos_ix is None and num_fields > 1:
        effective_pos_ix = ix_to_tag_list[0]

    per_field_predictions = []
    field_stats = []

    for local_idx in range(num_fields):
        if num_fields == 1:
            global_idx = field_index if field_index is not None else 0
        else:
            global_idx = local_idx
        field_name = field_names[local_idx]
        field_tag_to_ix = tag_dict_list[local_idx]
        field_ix_to_tag = ix_to_tag_list[local_idx]
        field_logits = all_field_logits[local_idx]
        gold_field = gold_per_field[local_idx]

        raw_acc = calculate_accuracy(field_logits, gold_field)
        logger.info(f"[test raw] {field_name} raw (D1) {raw_acc:.4f}")

        apply_mask = (
            mask_inference and legal_morph and global_idx is not None and global_idx > 0
        )
        filtered_logits = field_logits
        filt_stats = None

        if apply_mask:
            if not pos_list:
                logger.warning(f"Skipping legal-morph mask for {field_name}: missing POS sequences.")
            elif effective_pos_ix is None:
                logger.warning(f"Skipping legal-morph mask for {field_name}: missing POS vocabulary.")
            else:
                diag_ctr = new_diag()
                filtered_logits = apply_legal_mask(
                    predicted_sents=field_logits,
                    pos_list=pos_list,
                    field_tag_to_ix=field_tag_to_ix,
                    field_ix_to_tag=field_ix_to_tag,
                    pos_ix_to_tag=effective_pos_ix,
                    field_idx=global_idx,
                    field_name=field_name,
                    diag_ctr=diag_ctr
                )
                legal_rules_for_field = get_legal_rules_for_field(global_idx)
                _, filt_stats = calculate_accuracy_for_filtered_predictions(
                    filtered_logits,
                    gold_field,
                    pos_list=pos_list,
                    field_idx=global_idx,
                    field_tag_to_ix=field_tag_to_ix,
                    field_ix_to_tag=field_ix_to_tag,
                    pos_ix_to_tag=effective_pos_ix,
                    legal_rules_for_field=legal_rules_for_field,
                    raw_unmasked_scores=field_logits
                )
                if filt_stats is not None:
                    warn_if_filtered_lt_raw(filt_stats, "test")
                    log_filtered_policy_stats("test", field_name, filt_stats)
                    raw_at_d3_val = filt_stats.get('raw_at_d3')
                    raw_at_d3_str = f"{raw_at_d3_val:.4f}" if raw_at_d3_val is not None else "n/a"
                    logger.info(f"[test raw@D3] {field_name} raw@D3 {raw_at_d3_str}")
                    logger.info(f"[test filtered] {field_name} filtered (D3) {filt_stats['filtered_accuracy']:.4f}")

        predictions = logits_to_predictions(filtered_logits)
        per_field_predictions.append(predictions)
        field_stats.append(filt_stats)

    results = combine_field_predictions(per_field_predictions)
    shaped_results = results if return_shaped_results else None

    literal_test_tags = []
    for sent in test_field_tags:
        sent_literal = []
        for field_idx, field_tags in enumerate(sent):
            ix_map = ix_to_tag_list[field_idx if field_idx < len(ix_to_tag_list) else 0]
            if isinstance(ix_map, dict):
                field_literal = [ix_map.get(tag.item(), 'OOV') for tag in field_tags]
            else:
                field_literal = [ix_map[tag.item()] if tag.item() < len(ix_map) else 'OOV' for tag in field_tags]
            sent_literal.append(field_literal)
        literal_test_tags.append(sent_literal)

    literal_test_predicted = []
    for sent in results:
        sent_literal = []
        for field_idx, field_tags in enumerate(sent):
            ix_map = ix_to_tag_list[field_idx if field_idx < len(ix_to_tag_list) else 0]
            if isinstance(ix_map, dict):
                field_literal = [ix_map.get(tag, 'OOV') for tag in field_tags]
            else:
                field_literal = [ix_map[tag] if 0 <= tag < len(ix_map) else 'OOV' for tag in field_tags]
            sent_literal.append(field_literal)
        literal_test_predicted.append(sent_literal)

    write_predictions_to_file(results, test_sents, test_words, out_path + "-tagged.tsv",
                              ix_to_tag_list, ground_truth=literal_test_tags, field_names=field_names,
                              test_sent_sources=test_sent_sources)

    report_dicts = get_classification_report(literal_test_tags, literal_test_predicted, out_path, model_path,
                                            len(tag_dict_list), field_names=field_names)

    for fname, report_dict in zip(field_names, report_dicts):
        logger.info(f"Result {fname} precision: {report_dict['accuracy']}")
        logger.info(f"Result {fname} (weighted) recall: {report_dict['weighted avg']['recall']}")
        logger.info(f"Result {fname} (weighted) f1: {report_dict['weighted avg']['f1-score']}")
    if return_shaped_results:
        return report_dicts, shaped_results
    return report_dicts


def get_classification_report(test_tags, test_predicted, out_path, model_path, num_fields, field_names=None):
    if not field_names:
        field_names = [f"{i}" for i in range(num_fields)]

    outpaths = [out_path+f"-{field_name}.report" for field_name in field_names]
    report_dicts = []
    for field_idx, out_path in enumerate(outpaths):
        field_true = [tag for sent in test_tags for tag in sent[field_idx]]
        field_predicted = [tag for sent in test_predicted for tag in sent[field_idx]]
        report = classification_report(field_true, field_predicted)
        report_dict = classification_report(field_true, field_predicted, output_dict=True)
        report_dicts.append(report_dict)
        with open(out_path, 'w+', encoding='utf8') as report_file:
            report_file.write("Classification report:\n")
            report_file.write("Model: {}\n".format(model_path))
            report_file.write("-------------------------------------\n")
            report_file.write(report)
    return report_dicts


def write_predictions_to_file(results, sentences, test_words, out_path, tag_dict_list, ground_truth=None,
                              field_names=None, test_sent_sources=None):
    if not test_sent_sources:
        test_sent_sources = [("", "") for _ in sentences]
    num_fields = len(tag_dict_list)
    with open(out_path, 'w+', encoding='utf8', newline="") as out_f:
        tsv_writer = csv.writer(out_f, delimiter='\t')
        if ground_truth:
            column_names = ['source_file', 'source_sheet', 'sentence_id', 'word']
            if field_names:
                if len(field_names) == num_fields:
                    for name in field_names:
                        column_names.extend([f"true_{name}", f"predicted_{name}"])
                else:
                    logger.info("Please provide field names according to number of fields")
            else:
                for i in range(num_fields):
                    column_names.extend([f"true_{i}", f"predicted_{i}"])
            tsv_writer.writerow(column_names)
            for i, (sent_words, true_fields, pred_fields, source) in enumerate(zip(test_words, ground_truth,
                                                                                   results, test_sent_sources)):
                for j, word in enumerate(sent_words):
                    row = [source[0], source[1], i, word]
                    for field_idx, (field_true_tags, field_pred_tags) in enumerate(zip(true_fields, pred_fields)):
                        row.extend([field_true_tags[j], tag_dict_list[field_idx][field_pred_tags[j]]])
                    tsv_writer.writerow(row)
        else:
            column_names = ['source_file', 'source_sheet', 'sentence_id', 'word']
            if field_names:
                if len(field_names) == num_fields:
                    for name in field_names:
                        column_names.extend([f"predicted_{name}"])
                else:
                    logger.info("Please provide field names according to number of fields")
            else:
                for i in range(num_fields):
                    column_names.extend([f"predicted_{i}"])
            tsv_writer.writerow(column_names)
            for i, (sentence, pred_fields, source) in enumerate(zip(sentences, results, test_sent_sources)):
                for j, word in enumerate(sentence):
                    row = [source[0], source[1], i, word[0]]
                    for field_idx, field_pred_tags in enumerate(pred_fields):
                        row.extend([tag_dict_list[field_idx][field_pred_tags[j]]])
                    tsv_writer.writerow(row)


def tag(data_path, model_path, word_dict_path, char_dict_path,
        bpe_dict_path, tag_dict_path, word_emb_dim, char_emb_dim, hidden_dim, dropout,
        num_kernels, kernel_width, by_char=False, by_bpe=False,
        out_path=None, cnn=False, directions=1, device='cpu', morph=None, use_true_pos=False,
        legal_morph=False, mask_val=False):
    untagged_data, untagged_sent_objects = load_data.prepare_untagged_data(data_path)
    untagged_sents = [sent for sent in untagged_data if len(sent) > 0]

    assert len(untagged_sents) == len(untagged_sent_objects), "Length of sentences not the same!"

    model_path_parts = model_path.split(".")
    dict_path_parts = tag_dict_path.split(".")
    if morph:
        field_names = ["pos", "an1", "an2", "an3", "enc"]
    else:
        field_names = ["pos"]

    if device == torch.device('cpu'):
        map_location = device
    else:
        map_location = None

    word_dict = torch.load(word_dict_path)
    char_dict = torch.load(char_dict_path)
    bpe_dict = torch.load(bpe_dict_path)
    # tag_dict is a dictionary mapping tag to index!
    tag_dict_path_list = [dict_path_parts[0] + f"-{field_name}." + dict_path_parts[1]
                            for field_name in field_names]
    tag_dict_list = [torch.load(tag_dict_path) for tag_dict_path in tag_dict_path_list]
    ix_to_tag_list = [reverse_dict(tag_dict) for tag_dict in tag_dict_list]

    if DEBUG_MASK:
        for f, ix_to_tag in enumerate(ix_to_tag_list):
            expected = set(range(len(ix_to_tag)))
            assert expected == set(ix_to_tag.keys()), "ix_to_tag is missing indices"
    base_model = base_model_factory(by_char or by_bpe, cnn)

    if by_char:
        test_words = [prepare_sequence_for_chars(sent, word_dict, char_dict) for sent in untagged_sents]
    elif by_bpe:
        test_words = [prepare_sequence_for_bpes(sent, word_dict, bpe_dict) for sent in untagged_sents]
    else:
        test_words = [prepare_sequence_for_words(sent, word_dict).to(device=device) for sent in untagged_sents]

    if morph == MTL:
        model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout, len(word_dict),
                           len(char_dict) if by_char else len(bpe_dict), [len(tag_dict) for tag_dict in tag_dict_list],
                           num_kernels, kernel_width, directions=directions, device=device,
                           model_type=base_model)
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        model = model.to(device=device)
        tag_scores = predict_tags(model, test_words)
        pos_list = flatten_pos_sequences(
            [[sentence_scores[0].argmax(dim=1).detach().cpu()] if sentence_scores else [] for sentence_scores in tag_scores]
        )

        per_field_predictions = []
        for field_idx, field_name in enumerate(field_names):
            field_logits = extract_field_logits(tag_scores, field_idx)
            apply_mask = legal_morph and mask_val and field_idx > 0
            if apply_mask and pos_list:
                masked_logits = apply_legal_mask(
                    predicted_sents=field_logits,
                    pos_list=pos_list,
                    field_tag_to_ix=tag_dict_list[field_idx],
                    field_ix_to_tag=ix_to_tag_list[field_idx],
                    pos_ix_to_tag=ix_to_tag_list[0],
                    field_idx=field_idx,
                    field_name=field_name
                )
                logits_for_decode = masked_logits
            else:
                logits_for_decode = field_logits
            per_field_predictions.append(logits_to_predictions(logits_for_decode))

        results = combine_field_predictions(per_field_predictions)

    else:
        pos_model_path = model_path_parts[0] + f"-pos." + model_path_parts[1]
        pos_dict = tag_dict_list[0]
        model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout, len(word_dict),
                           len(char_dict) if by_char else len(bpe_dict), [len(pos_dict)],
                           num_kernels, kernel_width, directions=directions, device=device,
                           model_type=base_model)

        model.load_state_dict(torch.load(pos_model_path, map_location=map_location))
        model = model.to(device=device)

        tag_scores = predict_tags(model, test_words)

        pos_sentences = []
        for sent_idx, sentence_scores in enumerate(tag_scores):
            sentence_results = []
            for _, field_scores in enumerate(sentence_scores):
                field_results = []
                for word_idx, word_scores in enumerate(field_scores):
                    if legal_morph:
                        filter_invalid_pos_tags(ix_to_tag_list, field_results, untagged_sents,
                                                word_scores, sent_idx, word_idx)
                    else:
                        field_results.append(np.argmax(word_scores.cpu().detach().numpy()))
                sentence_results.append(field_results)
            pos_sentences.append(sentence_results)

        results_by_field = [pos_sentences]
        pos_sentence_preds = [sentence[0] if sentence else [] for sentence in pos_sentences]
        mask_pos_list = [torch.LongTensor(seq) for seq in pos_sentence_preds]

        if morph == FLAT or morph == HIERARCHICAL:

            pos_dict_size = 0
            mask_pos_for_fields = [tensor.detach().cpu() for tensor in mask_pos_list]

            if morph == HIERARCHICAL:
                if use_true_pos:
                    # Be very sure the words have POS tags;
                    # otherwise you'll be using a whole lot of Nones for prediction
                    test_pos = [[word_an.pos for word_an in sent_obj.word_analyses]
                                for sent_obj in untagged_sent_objects]
                    test_pos = [torch.LongTensor([get_index(pos, pos_dict) for pos in sent_poses]).to(device=device) for sent_poses in test_pos]
                    mask_pos_for_fields = [tensor.detach().cpu() for tensor in test_pos]
                else:
                    test_pos = [torch.LongTensor(sentence).to(device=device) for sentence in pos_sentence_preds]
                    mask_pos_for_fields = [tensor.detach().cpu() for tensor in test_pos]

                if by_bpe or by_char:
                    test_words = [[(idxs[0], idxs[1], tag_idx) for idxs, tag_idx in zip(sentence, sent_tags)]
                                  for sentence, sent_tags in zip(test_words, test_pos)]
                else:
                    test_words = [[(word_idx, tag_idx) for word_idx, tag_idx in zip(sentence, sent_tags)]
                                  for sentence, sent_tags in zip(test_words, test_pos)]
                pos_dict_size = len(tag_dict_list[0])

            for field_idx, field_name in enumerate(field_names[1:], start=1):
                field_model_path = model_path_parts[0] + f"-{field_name}." + model_path_parts[1]
                model = MTLWrapper(word_emb_dim, char_emb_dim, hidden_dim, dropout, len(word_dict),
                                   len(char_dict) if by_char else len(bpe_dict), [len(tag_dict_list[field_idx])],
                                   num_kernels, kernel_width, directions=directions, device=device,
                                   model_type=base_model, pos_dict_size=pos_dict_size)

                model.load_state_dict(torch.load(field_model_path, map_location=map_location))
                model = model.to(device=device)

                tag_scores = predict_tags(model, test_words)

                field_logits = extract_field_logits(tag_scores, 0)
                apply_mask = legal_morph and mask_val and mask_pos_for_fields
                if apply_mask:
                    masked_logits = apply_legal_mask(
                        predicted_sents=field_logits,
                        pos_list=mask_pos_for_fields,
                        field_tag_to_ix=tag_dict_list[field_idx],
                        field_ix_to_tag=ix_to_tag_list[field_idx],
                        pos_ix_to_tag=ix_to_tag_list[0],
                        field_idx=field_idx,
                        field_name=field_name
                    )
                    logits_for_decode = masked_logits
                else:
                    logits_for_decode = field_logits

                field_predictions = logits_to_predictions(logits_for_decode)
                wrapped_predictions = [[sentence_preds] for sentence_preds in field_predictions]
                results_by_field.append(wrapped_predictions)

            results = reshape_by_field_to_by_sent(results_by_field, num_fields=len(field_names))
        else:
            # this is POS only tagging
            results = reshape_by_field_to_by_sent(results_by_field, num_fields=1)

    updated_sentences = add_tags_to_sent_objs(untagged_sent_objects, results, ix_to_tag_list, field_names)
    write_tagged_sents(updated_sentences, out_path)


def reshape_by_field_to_by_sent(results_by_field, num_fields=5):
    """
    Input is list of length num_fields, where each inner list is of sentences, with num words
    Output needs to be list of length num_sentences, where len(output[i]) == num_fields
    :param results_by_field:
    :param num_fields:
    :return:
    """
    sentences = []
    for sentence_idx in range(len(results_by_field[0])):
        sentence_words = []
        for field_idx in range(num_fields):
            if results_by_field[field_idx][sentence_idx]:
                sentence_words.append(results_by_field[field_idx][sentence_idx][0])
            else:
                # in case the sentence is invalid
                sentence_words.append([])
        sentences.append(sentence_words)
    return sentences


def add_tags_to_sent_objs(sentences, tags, ix_to_tag_list, field_names):
    """
    Shape of tags:
    len(tags) == len(sentences)
    len(tags[i]) == len(field_names) == len(ix_to_tag_list)
    len(tags[i][j]) == len(field_names)
    :param sentences:
    :param tags:
    :param ix_to_tag_list:
    :param field_names:
    :return:
    """
    field_to_column_dict = {"pos": "pos", "an1": "analysis1", "an2": "analysis2",
                            "an3": "analysis3", "enc": "enclitic_pronoun"}
    for sentence, sent_tags in zip(sentences, tags):
        for field_name, field_tags, ix_to_tag in zip(field_names, sent_tags, ix_to_tag_list):
            for word_an, tag in zip(sentence.word_analyses, field_tags):
                word_an.set_val(field_to_column_dict[field_name], ix_to_tag[tag])
    return sentences


def write_tagged_sents(sentences, dir):
    write_sentences_to_excel(sentences, dir)


def kfold_val(data_paths, model_path, word_dict_path, char_dict_path, bpe_path,
              tag_dict_path, result_path, k, word_emb, char_emb, hidden_dim,
              dropout, num_kernels=1000, kernel_width=6, by_char=False, by_bpe=False,
              with_smoothing=False, cnn=False, directions=1, device='cpu',
              epochs=300, morph=None, weight_decay=0, use_true_pos=False, loss_weights=(1,1,1,1,1), legal_morph=False):
    logger.info("Beginning k-fold validation")
    results = []
    fold = 0
    for train_sentences, val_sentences, test_sentences, train_word_count in \
            load_data.prepare_kfold_data(data_paths, k=k, seed=0):
        logger.info("Beginning fold #{}".format(fold+1))
        new_model_path = add_fold_dir_to_path(model_path, fold)
        new_word_path = add_fold_dir_to_path(word_dict_path, fold)
        new_char_path = add_fold_dir_to_path(char_dict_path, fold)
        new_tag_path = add_fold_dir_to_path(tag_dict_path, fold)
        new_bpe_path = add_fold_dir_to_path(bpe_path, fold)

        train(train_sentences, val_sentences, new_model_path, new_word_path,
              new_char_path, new_bpe_path, new_tag_path, train_word_count, word_emb,
              char_emb, hidden_dim, dropout, num_kernels, kernel_width, by_char, by_bpe,
              with_smoothing, cnn, directions, device, epochs=epochs, morph=morph,
              weight_decay=weight_decay, loss_weights=loss_weights, legal_morph=legal_morph)

        new_result_path = add_fold_dir_to_path(result_path, fold)
        results.append(test(test_sentences, new_model_path, new_word_path, new_char_path, new_bpe_path,
                            new_tag_path, word_emb, char_emb, hidden_dim, dropout, num_kernels,
                            kernel_width, by_char, by_bpe, out_path=new_result_path,
                            cnn=cnn, directions=directions, device=device, morph=morph, use_true_pos=use_true_pos,
                            enforce_legal_morphology=legal_morph, mask_val=legal_morph))

        fold += 1
    agg_result_path = result_path + ".agg_res"
    mic_prec = []
    mac_prec = []
    weight_prec = []
    with open(agg_result_path, 'w+') as f:
        if not morph:
            for i, res in enumerate(results):
                micro = res.get("micro avg", res["accuracy"])
                f.write("Fold {}:\nmicro avg: {}\nmacro avg: {}\nweighted avg: {}\n".format(
                    i, micro, res["macro avg"], res["weighted avg"]))
                if "micro avg" in res:
                    mic_prec.append(res["micro avg"]["precision"])
                else:
                    mic_prec.append(res["accuracy"])  # TODO you need to check how this appears in the dict!
                mac_prec.append(res["macro avg"]["precision"])
                weight_prec.append(res["weighted avg"]["precision"])
            avg = np.mean(mic_prec)
            std = np.std(mic_prec)
            f.write("------------\nMicro:\nAverage precision: {}\nStandard deviation:{}\n".format(avg, std))
            avg = np.mean(mac_prec)
            std = np.std(mac_prec)
            f.write("------------\nMacro:\nAverage precision: {}\nStandard deviation:{}\n".format(avg, std))
            avg = np.mean(weight_prec)
            std = np.std(weight_prec)
            f.write("------------\nWeighted:\nAverage precision: {}\nStandard deviation:{}\n".format(avg, std))
        else:
            an_names = ["pos", "analysis1", "analysis2",
                        "analysis3", "enclitic"]
            for i, res in enumerate(results):
                # if isinstance(res[0], list):
                #     res = [an_res[0] for an_res in res]
                f.write(f"Fold {i} results:\n")
                for an_field, an_res in zip(an_names, res):
                    if isinstance(an_res, list):
                        an_res = an_res[0]
                    micro = an_res.get("micro avg", an_res["accuracy"])
                    f.write(f"{an_field}\nmicro: {micro}\nmacro: {an_res['macro avg']}"
                            f"\nweighted{an_res['weighted avg']}\n")
                try:
                    mic_prec.append([an_res["micro avg"]["precision"] for an_res in res])
                except KeyError:
                    mic_prec.append([an_res["accuracy"] for an_res in res])
                mac_prec.append([an_res["macro avg"]["precision"] for an_res in res])
                weight_prec.append([an_res["weighted avg"]["precision"] for an_res in res])
            avg = [np.mean([fold_res[i] for fold_res in mic_prec]) for i in range(len(an_names))]
            std = [np.std([fold_res[i] for fold_res in mic_prec]) for i in range(len(an_names))]
            for an_field, a, s in zip(an_names, avg, std):
                f.write("------------\nMicro {}:\nAverage precision: {}\nStandard deviation:{}\n".
                        format(an_field, a, s))
            avg = [np.mean([fold_res[i] for fold_res in mac_prec]) for i in range(len(an_names))]
            std = [np.std([fold_res[i] for fold_res in mac_prec]) for i in range(len(an_names))]
            for an_field, a, s in zip(an_names, avg, std):
                f.write("------------\nMacro {}:\nAverage precision: {}\nStandard deviation:{}\n".
                        format(an_field, a, s))
            avg = [np.mean([fold_res[i] for fold_res in weight_prec]) for i in range(len(an_names))]
            std = [np.std([fold_res[i] for fold_res in weight_prec]) for i in range(len(an_names))]
            for an_field, a, s in zip(an_names, avg, std):
                f.write("------------\nWeighted {}:\nAverage precision: {}\nStandard deviation:{}\n".
                        format(an_field, a, s))

    return results


def add_fold_dir_to_path(path, fold_num):
    split_path = os.path.split(path)
    new_dir = os.path.join(split_path[0], "fold_{}".format(fold_num))
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    new_path = os.path.join(new_dir, split_path[1])
    return new_path


def main(args):
    today = str(datetime.date.today())

    log_name = args.log_file if args.log_file else f"{today}.log"
    split_path = os.path.split(log_name)
    # if not os.path.exists(split_path[0]):
    #     os.mkdir(split_path[0])
    char_based = args.char_based
    bpe_based = args.bpe_based
    smoothed = True if args.smoothed else False
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    model_path = os.path.join(args.model_dir, args.model_name)
    word_dict_path = os.path.join(args.model_dir, args.word_dict_name)
    char_dict_path = os.path.join(args.model_dir, args.char_dict_name)
    bpe_dict_path = os.path.join(args.model_dir, args.bpe_dict_name)
    tag_dict_path = os.path.join(args.model_dir, args.tag_dict_name)

    if args.result_path:
        rp_base_dir = os.path.split(args.result_path)[0]
        if not os.path.isdir(rp_base_dir):
            os.mkdir(rp_base_dir)

    morph = None
    if args.morph:
        if args.flat:
            morph = FLAT
        elif args.multitask:
            morph = MTL
        elif args.hierarchical:
            morph = HIERARCHICAL

    if args.train:
        if args.no_val:
            train_data, frequencies = load_data.prepare_train_data(args.data_paths)
            val_data = None
        else:
            _, _, train_path, val_path = split_train_val(args.data_paths, args.model_dir, max_words=args.max_words)
            train_data, frequencies = load_data.prepare_train_data([train_path])
            val_data, _ = load_data.prepare_test_data([val_path], sources=False)

        train(train_data, val_data, model_path, word_dict_path,
              char_dict_path, bpe_dict_path, tag_dict_path, frequencies,
              word_emb_dim=args.word_emb_dim, char_emb_dim=args.char_emb_dim,
              hidden_dim=args.hidden_dim, dropout=args.dropout, num_kernels=args.num_kernels,
              kernel_width=args.kernel_width, with_smoothing=smoothed,
              by_char=char_based, by_bpe=bpe_based, cnn=args.cnn, directions=args.directions,
              device=args.device, epochs=args.epochs, lr=args.learning_rate,
              batch_size=args.batch_size, morph=morph, loss_weights=args.loss_weights, seed=args.seed,
              legal_morph=args.legal_morph)
    elif args.test:
        test_data, sources = load_data.prepare_test_data(args.data_paths, out_dir=args.model_dir, sources=True)

        test(test_data, model_path, word_dict_path,
             char_dict_path, bpe_dict_path, tag_dict_path,
             word_emb_dim=args.word_emb_dim, char_emb_dim=args.char_emb_dim,
             hidden_dim=args.hidden_dim, dropout=args.dropout,
             num_kernels=args.num_kernels, kernel_width=args.kernel_width,
             by_char=char_based, by_bpe=bpe_based, cnn=args.cnn, directions=args.directions,
             out_path=args.result_path, device=args.device, morph=morph, use_true_pos=args.use_true_pos,
             test_sent_sources=sources, enforce_legal_morphology=args.legal_morph,
             mask_val=args.mask_val)

    elif args.tag:
        tag(args.data_paths, model_path, word_dict_path, char_dict_path, bpe_dict_path,
            tag_dict_path, word_emb_dim=args.word_emb_dim, char_emb_dim=args.char_emb_dim,
            hidden_dim=args.hidden_dim, dropout=args.dropout,
            num_kernels=args.num_kernels, kernel_width=args.kernel_width,
            by_char=char_based, by_bpe=bpe_based, cnn=args.cnn, directions=args.directions,
            out_path=args.result_path, device=args.device, morph=morph, use_true_pos=args.use_true_pos,
            legal_morph=args.legal_morph, mask_val=args.mask_val)

    elif args.kfold_validation:
        kfold_val(args.data_paths, model_path, word_dict_path,
                  char_dict_path, bpe_dict_path, tag_dict_path,
                  args.result_path, args.k, args.word_emb_dim, args.char_emb_dim,
                  args.hidden_dim, args.dropout, args.num_kernels, args.kernel_width,
                  char_based, bpe_based, cnn=args.cnn, directions=args.directions,
                  device=args.device, epochs=args.epochs, morph=morph, use_true_pos=args.use_true_pos,
                  loss_weights=args.loss_weights, legal_morph=args.legal_morph)
    else:
        print("Must select either train (-r), test (-e) or tag (-a)")


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Train, test, or calculate POS')
    parser.add_argument(
        'model_dir',
        help="Path to directory to save or load the model and all its necessary dicts")
    parser.add_argument(
        '--model_name', help="Name for model", default="ja.pt")
    parser.add_argument(
        '--word_dict_name', help="Name for word dictionary", default='words.dct')
    parser.add_argument(
        '--char_dict_name', help="Name for char dictionary", default="char.dct")
    parser.add_argument(
        '--bpe_dict_name', help="Name for BPE dictionary", default='bpe.dct')
    parser.add_argument(
        '--tag_dict_name', help="Name for POS tag dictionary", default='tags.dct')
    parser.add_argument(
        '--no_val',
        action='store_true',
        help="For training a model on all the data, without early stopping")

    parser.add_argument('data_paths', nargs='+', help="Path to data pickles to train on/test on/tag")

    parser.add_argument('--morph', action='store_true', help="Tag morphological analyses")

    morph_group = parser.add_mutually_exclusive_group()

    morph_group.add_argument(
        '--flat',
        action='store_true',
        help="Train separate model for each analysis field")
    morph_group.add_argument(
        '--multitask',
        action='store_true',
        help="Train single multitask model for POS and all morphological analyses")
    morph_group.add_argument(
        '--hierarchical',
        action='store_true',
        help="Train hierarchical models for morphological analyses")

    parser.add_argument('-we', '--word_emb_dim', type=int, default=100, help="Word embedding dimensionality")
    parser.add_argument('-ce', '--char_emb_dim', type=int, default=25,
                        help="Character (or BPE) embedding dimensionality")
    parser.add_argument('-he', '--hidden_dim', type=int, default=100, help="Hidden state dimensionality")

    parser.add_argument('--cnn', action='store_true', help="Use this flag to use char CNN instead of char LSTM")
    parser.add_argument('-nk', '--num_kernels', type=int, default=500, help="Number of kernels to use for char-CNN")
    parser.add_argument('-kw', '--kernel_width', type=int, default=6, help="Kernel width to use for char-CNN")

    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate in LSTM")
    parser.add_argument('--directions', type=int, default=2, choices=[1, 2], help="Number of directions in LSTM")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help="Learning rate")
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help="Size of batch")

    model_type = parser.add_mutually_exclusive_group(required=True)

    model_type.add_argument('-w', '--word_based', action='store_true')
    model_type.add_argument('-c', '--char_based', action='store_true')
    model_type.add_argument('-b', '--bpe_based', action='store_true')

    parser.add_argument('-s', '--smoothed', action='store_true')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-r', '--train', action='store_true')
    group.add_argument('-e', '--test', action='store_true')
    group.add_argument('-a', '--tag', action='store_true')
    group.add_argument('-kfv', '--kfold_validation', action='store_true')

    parser.add_argument('-k', type=int, default=10, help="Number of folds for kfold validation")

    parser.add_argument('--epochs', type=int, default=30, help="Maximum number of epochs to train")

    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')

    parser.add_argument('-l', '--log_file', type=str, help="Path to save log file at")

    parser.add_argument('-rp', '--result_path', type=str, default="",
                        help="Path *without extension* to save test/tag results at")
    parser.add_argument('--use_true_pos',
                        action='store_true',
                        help="When testing hierarchical models, use true pos tags and not predicted pos tags")

    parser.add_argument('--testing', action='store_true',
                        help="Prints command line arguments for this run, and exits program")
    parser.add_argument('--debug', action='store_true', help='Set logger level to debug')

    parser.add_argument('--add_path', help='Append script dir to python path')

    parser.add_argument('--loss_weights',
                        nargs=5,
                        type=int,
                        default=[1, 1, 1, 1, 1],
                        help="Weights for averaging loss in MTL leaning, in order: POS, an1, an2, an3, enc")
    parser.add_argument('--seed', type=int, default=42, help="Use specific seed for random elements in training")

    parser.add_argument('--max_words',
                        type=int,
                        default=math.inf,
                        help='For training on smaller (random) subset of input sentences')
    parser.add_argument('--legal_morph',
                        action='store_true',
                        help='Enforce legal morphological values according to constrains in legal_values.json')
    parser.add_argument('--mask_train', action='store_true', default=False,
                        help='Apply legal-morph masking during training (before loss).')
    parser.add_argument('--mask_val', action='store_true', default=False,
                        help='Apply legal-morph masking during validation/inference.')
    parser.add_argument('--treat_gold_illegal_as_underscore', action='store_true', default=True,
                        help='When gold is illegal or None under legal-morph, treat "_" as the correct filtered label.')
    parser.add_argument('--verbose_mask_debug', action='store_true',
                        help='Print detailed masking diagnostics each epoch.')

    args, unknown = parser.parse_known_args()

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # if legal_morph is True and user didn't pass either flag, set mask_val=True
    if getattr(args, 'legal_morph', False):
        if not any([getattr(args, 'mask_train', False), getattr(args, 'mask_val', False)]):
            args.mask_val = True    # val-only masking, by default

    set_verbose_mask_debug(getattr(args, 'verbose_mask_debug', False))

    # quiet the RNN dropout warning for single-layer models
    if hasattr(args, "dropout") and args.dropout > 0:
        # if using a single-layer model (default for most architectures)
        print("[note] Setting dropout=0 to avoid ineffective-dropout warning for single-layer models.")
        args.dropout = 0.0

    if args.debug:
        logger.setLevel(level=logging.DEBUG)

    if args.add_path:
        import sys
        sys.path.append(args.add_path)
        import load_data
        from utils import split_train_val
        from data_classes import write_sentences_to_excel

    if args.testing:
        print(args)
    else:
        if args.legal_morph:
            # if enforcing legal morphological values, load legal_values.json handed globally
            with open("legal_values.json", encoding='utf-8') as legal_values_file:
                legal_values = json.load(legal_values_file)

        # logger for mask warnings
        mask_logger = logging.getLogger('mask_warnings')
        mask_logger.setLevel(logging.INFO)
        mask_handler = logging.FileHandler('mask_warnings.log', mode='a', encoding='utf-8')
        mask_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        if not mask_logger.hasHandlers():
            mask_logger.addHandler(mask_handler)

        main(args)
