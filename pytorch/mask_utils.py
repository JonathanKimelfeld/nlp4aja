#   Helpers for the legal-morph mask and its associated statistics

import torch
import json
import os
import numpy as np
from collections import Counter
from debug_run import *

# Constants
NEG = -1e9  # safer than -inf for masking with LogSoftmax

# Global legal_values
_legal_values = None

VERBOSE_MASK_DEBUG = False
MASK_DEBUG_CHECK = False


def set_verbose_mask_debug(enabled):
    """Toggle verbose mask diagnostics."""
    global VERBOSE_MASK_DEBUG
    VERBOSE_MASK_DEBUG = bool(enabled)


"""
Unconstrained vs. constrained rules:
    - RULELESS / UNCONSTRAINED: no rule exists for field. contribute to D1 (TOTAL) but not D2 or D3
    - HAS RULE / CONSTRAINED: a non-empty rule list (not ["NA"])
    - ENC rules are not POS conditioned. "NA" is valid and means no enclitic

Filtering strategy:
    - D1 counts every token
    - D2 keeps only constrained ones
    - D3 keeps the constrained tokens whose gold tags are non-null and legal
    - raw@D3 evaluates pre-masked accuracy on D3 so it is comparable to filtered accuracy (also on D3)
    - Unconstrained and null-gold tokens never enter D3.
"""


def rulescan_for_field(field_idx, val_pos_list, ix_to_tag_list, legal_rules_for_field):
    """Print TOTAL / UNCONSTRAINED / HAS_RULE counts for one field."""
    if legal_rules_for_field is None:
        print(f"[rulescan field={field_idx}] rules=None")
        return

    pos_ix_to_tag, field_ix_to_tag = ix_to_tag_list
    stats = {"TOTAL": 0, "UNCONSTRAINED": 0, "HAS_RULE": 0}

    if field_idx == 4:
        total_tokens = sum(
            len(seq[0]) if hasattr(seq[0], "__len__") else 1
            for seq in val_pos_list
        )
        if is_rule_unconstrained(legal_rules_for_field):
            stats["UNCONSTRAINED"] = total_tokens
        else:
            stats["HAS_RULE"] = total_tokens
        stats["TOTAL"] = total_tokens
    else:
        for seq in val_pos_list:
            pos_seq = seq[0].tolist() if hasattr(seq[0], "tolist") else seq[0]
            for pos_idx in pos_seq:
                if isinstance(pos_ix_to_tag, dict):
                    pos_tag_str = pos_ix_to_tag.get(pos_idx, f"POS_{pos_idx}")
                elif isinstance(pos_ix_to_tag, (list, tuple)) and 0 <= pos_idx < len(pos_ix_to_tag):
                    pos_tag_str = pos_ix_to_tag[pos_idx]
                else:
                    pos_tag_str = f"POS_{pos_idx}"

                pos_rules = legal_rules_for_field.get(pos_tag_str) if isinstance(legal_rules_for_field, dict) else None
                rule = pos_rules.get(analysis_key_for_field(field_idx)) if pos_rules else None

                if is_rule_unconstrained(rule):
                    stats["UNCONSTRAINED"] += 1
                else:
                    stats["HAS_RULE"] += 1

                stats["TOTAL"] += 1

    print(f"[rulescan field={field_idx}] {stats}")


def analysis_key_for_field(field_idx):
    if field_idx == 1:
        return "analysis1"
    if field_idx == 2:
        return "analysis2"
    if field_idx == 3:
        return "analysis3"
    if field_idx == 4:
        return "enclitic"
    return None


def is_rule_unconstrained(rule):
    """True if the rule supplies no usable tags."""
    if rule is None:
        return True
    if rule == "NA":
        return True
    if isinstance(rule, list) and (len(rule) == 0 or rule == ["NA"]):
        return True
    if isinstance(rule, (set, tuple)) and len(rule) == 0:
        return True
    return False


def load_legal_values():
    global _legal_values
    if _legal_values is None:
        legal_values_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "legal_values.json")
        if not os.path.exists(legal_values_path):
            legal_values_path = "../../nlpTaja/legal_values.json"
        
        if os.path.exists(legal_values_path):
            with open(legal_values_path, 'r', encoding='utf-8') as f:
                _legal_values = json.load(f)
        else:
            _legal_values = {}
    return _legal_values


def get_legal_rules_for_field(field_idx):
    """Get the rule table for a field, POS-conditioned or just the global enclitic rule list"""
    legal_values = load_legal_values()
    if field_idx == 4:
        return legal_values.get("enclitic", {})
    else:
        return legal_values.get("pos", {})


def is_rule_unconstrained(rule):
    """True if the rule carries no usable tags."""
    if rule is None:
        return True
    if rule == "NA":
        return True
    if isinstance(rule, list) and (len(rule) == 0 or rule == ["NA"]):
        return True
    if isinstance(rule, (set, tuple)) and len(rule) == 0:
        return True
    return False


def mask_tags_1d(tags_1d, allowed_ix_set):
    """Zero out disallowed tags by writing NEG into them."""
    if allowed_ix_set is None:
        return tags_1d
    mask = torch.ones_like(tags_1d, dtype=torch.bool)
    if allowed_ix_set:
        idx = torch.tensor(sorted(allowed_ix_set), device=tags_1d.device, dtype=torch.long)
        mask[idx] = False
    return tags_1d.masked_fill(mask, NEG)


def is_gold_illegal_or_none(gold_tag_str, pos_label, field_idx, legal_rules):
    """True when gold is missing or filtered out according to legal_values.json"""
    if gold_tag_str is None or gold_tag_str == "_":
        return True

    analysis_key = analysis_key_for_field(field_idx)

    if field_idx < 4:
        pos_rules = legal_rules.get(pos_label, None)
        if pos_rules is None:
            return False
        legal_rules_for_field = pos_rules.get(analysis_key, None)
    else:
        legal_rules_for_field = legal_rules

    # no rule exists or rule is None/"NA" -> not illegal
    if not legal_rules_for_field or legal_rules_for_field == "NA":
        return False

    # If gold_tag_str is not in the rule set, then it's illegal
    return gold_tag_str not in legal_rules_for_field


def _pos_to_str(pos_idx, pos_ix_to_tag):
    """Map a POS index to its tag."""
    if isinstance(pos_ix_to_tag, dict):
        return pos_ix_to_tag.get(pos_idx, f"POS_{pos_idx}")
    elif isinstance(pos_ix_to_tag, (list, tuple)) and 0 <= pos_idx < len(pos_ix_to_tag):
        return pos_ix_to_tag[pos_idx]
    return f"POS_{pos_idx}"


def _gold_to_str(gold_idx, field_ix_to_tag):
    """Map a gold index to its textual tag."""
    if isinstance(field_ix_to_tag, dict):
        return field_ix_to_tag.get(gold_idx, f"TAG_{gold_idx}")
    elif isinstance(field_ix_to_tag, (list, tuple)) and 0 <= gold_idx < len(field_ix_to_tag):
        return field_ix_to_tag[gold_idx]
    return f"TAG_{gold_idx}"


def normalize_logits_to_list_list(logits):
    """Ensure logits are shaped as list[list[tensor]]."""
    normalized = []
    for sent_logits in logits:
        if isinstance(sent_logits, list):
            normalized.append(sent_logits)
        else:
            normalized.append([sent_logits])
    return normalized


def lookup_rule(field_idx, pos_str, legal_values):
    """
    - if field is POS-conditioned, look up rule in legal_values["pos"][pos_str][analysisX]
    - if field is enclitic, look up rule in legal_values["enclitic"]
    """
    analysis_key = analysis_key_for_field(field_idx)
    if analysis_key is None:
        return None

    if analysis_key == "enclitic":
        rule = legal_values.get("enclitic", None)
    else:
        pos_rules = legal_values.get("pos", {}).get(pos_str, {}) if isinstance(legal_values.get("pos", {}), dict) else {}
        rule = pos_rules.get(analysis_key, None)
    return rule

def is_illegal(gold):
    """Return True when gold is a placeholder such as '_'."""
    return gold in {"_", "UNK"} or (isinstance(gold, str) and gold.startswith("UNK_"))


def get_mask_and_loss_decision(field_idx, pos_str, gold, legal_values=None, treat_illegal_as_underscore=True):
    """
    Decide how to treat a gold tag for loss masking/training.
    - if gold is "_" or None, skip loss
    - if rule is not None and gold is in rule, return rule and "train"
    - if rule is not None and gold is not in rule, return rule and "skip_loss"
    - if rule is None, return None and "train"
    """
    if legal_values is None:
        legal_values = load_legal_values()
    rule = lookup_rule(field_idx, pos_str, legal_values)
    if gold in {"_", None}:
        return rule, "skip_loss"
    if rule:
        if gold in rule:  # legal gold
            return rule, "train"
        else:  # illegal gold
            return rule, "skip_loss" if treat_illegal_as_underscore else "train"
    else:
        # no rule
        if is_illegal(gold):
            return None, "skip_loss" if treat_illegal_as_underscore else "train"
        else:  # legal but no rule known
            return None, "train"


def filter_invalid_pos_tags(ix_to_tag_list, field_results, untagged_sents, word_scores, sent_idx, word_idx):
    """Fallback POS selection when the best tag is "_" on a real word."""

    if ix_to_tag_list[0][np.argmax(word_scores.cpu().detach().numpy())] != "_":
        field_results.append(np.argmax(word_scores.cpu().detach().numpy()))

    else:
        word_text = untagged_sents[sent_idx][word_idx][0].strip()
        if word_text.isdigit() or len(word_text) < 3:
            field_results.append(np.argmax(word_scores.cpu().detach().numpy()))
        else:
            sorted_indices = np.argsort(word_scores.cpu().detach().numpy())
            field_results.append(sorted_indices[1])


def filter_invalid_analysis_tags(
        field_scores, field_idx, pos_tags_for_sent,
        field_tag_to_ix, field_ix_to_tag, pos_ix_to_tag, field_name,
        gold_field=None, metric_only=False, diag_ctr=None, sent_idx=None, is_training=False
):
    """Apply legal morph mask for a single sentence/field tensor."""
    legal_values = load_legal_values()
    masked = []

    for word_idx in range(field_scores.shape[1]):
        word_scores = field_scores[:, word_idx]
        pos_idx = pos_tags_for_sent[word_idx].item() if hasattr(pos_tags_for_sent[word_idx], "item") else pos_tags_for_sent[word_idx]
        pos_value = _pos_to_str(pos_idx, pos_ix_to_tag)
        analysis_key = analysis_key_for_field(field_idx)

        if analysis_key == "enclitic":
            rule = legal_values.get("enclitic")
            if isinstance(rule, dict):
                try:
                    mask_logger.warning(f"ENC rules should be a list, not dict. Got: {type(rule)}. Treating as unconstrained.")
                except NameError:
                    pass
                rule = None
        else:
            pos_rules = legal_values.get("pos", {}).get(pos_value, {}) if isinstance(legal_values.get("pos", {}), dict) else {}
            rule = pos_rules.get(analysis_key)

        if analysis_key == "enclitic" and gold_field is not None:
            gold_id = gold_field[word_idx].item() if hasattr(gold_field[word_idx], "item") else gold_field[word_idx]
            if 0 <= gold_id < len(field_ix_to_tag):
                gold_tag_str = field_ix_to_tag[gold_id]
                if gold_tag_str == "NA":
                    if isinstance(diag_ctr, dict):
                        diag_ctr["enc_gold_na_skip"] = diag_ctr.get("enc_gold_na_skip", 0) + 1
                    masked.append(word_scores)
                    continue

        if is_rule_unconstrained(rule):
            masked_word = word_scores
        else:
            allowed_tags = set(t for t in rule if t in field_tag_to_ix)
            if analysis_key == "enclitic":
                if "NA" in field_tag_to_ix:
                    allowed_tags.add("NA")
                if "_" in field_tag_to_ix:
                    allowed_tags.add("_")

            if not allowed_tags:
                if analysis_key != "enclitic" and "_" in field_tag_to_ix:
                    allowed_tags = {"_"}
                else:
                    masked.append(word_scores)
                    continue

            allowed_ix = {field_tag_to_ix[t] for t in allowed_tags if t in field_tag_to_ix}
            masked_word = mask_tags_1d(word_scores, allowed_ix)

        if not is_training and gold_field is not None:
            gold_id = gold_field[word_idx].item() if hasattr(gold_field[word_idx], "item") else gold_field[word_idx]
            if 0 <= gold_id < len(field_ix_to_tag):
                gold_tag_str = field_ix_to_tag[gold_id]
                if not is_rule_unconstrained(rule) and gold_tag_str not in rule:
                    try:
                        mask_logger.info(
                            f"Gold tag {gold_tag_str} not in legal set for field {field_idx} ({analysis_key}), POS {pos_value}, sent {sent_idx}, word {word_idx}"
                        )
                    except NameError:
                        pass

        masked.append(masked_word)

    return torch.stack(masked, dim=1)


def apply_legal_mask(
        predicted_sents,
        pos_list,
        field_tag_to_ix,
        field_ix_to_tag,
        pos_ix_to_tag,
        field_idx,
        field_name,
        diag_ctr=None
):
    """Apply filter_invalid_analysis_tags to each sentence and field."""
    masked_all = []

    for sent_idx, sentence_scores in enumerate(predicted_sents):
        pos_tags_for_sent = pos_list[sent_idx]
        sentence_fields = []

        # Handle both single tensor and list of tensors
        if isinstance(sentence_scores, list):
            # used in MTL mode: sentence_scores is [field1_tags, field2_tags, ...]
            if field_idx < len(sentence_scores):
                field_scores = sentence_scores[field_idx]
            else:
                field_scores = sentence_scores[0]
        else:
            # sentence_scores is a single tensor
            field_scores = sentence_scores

        # Ensure scores are a proper 2-D tensor before transposing
        field_scores = ensure_2d_tensor(field_scores)

        masked_field = filter_invalid_analysis_tags(
            field_scores.T,
            field_idx,
            pos_tags_for_sent,
            field_tag_to_ix,
            field_ix_to_tag,
            pos_ix_to_tag,
            field_name,
            metric_only=True,
            diag_ctr=diag_ctr,
            sent_idx=sent_idx,
        )
        sentence_fields.append(masked_field.T)  # filter_invalid_analysis_tags returns [V,T], transpose back to [T,V]

        masked_all.append(sentence_fields)

    return masked_all


def calculate_accuracy_for_filtered_predictions(predicted_tag_scores, true_tags_2d, pos_list=None,
                                                field_idx=None, field_tag_to_ix=None, field_ix_to_tag=None,
                                                pos_ix_to_tag=None, legal_rules_for_field=None,
                                                raw_unmasked_scores=None):
    """Compute filtered accuracy + raw@D3 stats (Exclude filtering strategy)."""
    totals = {
        "correct": 0,
        "raw_correct_d3": 0,
        "D1": 0,
        "D2": 0,
        "D3": 0,
        "masked_tokens": 0,
        "unconstrained_tokens": 0,
        "null_gold": 0,
        "gold_illegal": 0,
    }

    can_compute = all(
        obj is not None
        for obj in (pos_list, field_idx, field_tag_to_ix, field_ix_to_tag, pos_ix_to_tag, legal_rules_for_field)
    )

    def lookup_rule_for_pos(pos_str, f_idx):
        key = analysis_key_for_field(f_idx)
        if f_idx < 4:
            pos_rules = legal_rules_for_field.get(pos_str, {}) if isinstance(legal_rules_for_field, dict) else {}
            return pos_rules.get(key)
        return legal_rules_for_field if isinstance(legal_rules_for_field, (list, tuple, set)) else None

    def rule_to_allowed_set(rule_obj, analysis_idx):
        """Convert rule object to a set of allowed tags."""
        if isinstance(rule_obj, (list, tuple, set)):
            allowed = {tag for tag in rule_obj if tag is not None}
        elif rule_obj is None:
            allowed = set()
        else:
            allowed = {rule_obj}
        if analysis_idx == 4:
            allowed.update({"_", "NA"})
        return allowed

    illegal_prediction_violations = []

    for sent_idx, (pred_fields, true_fields) in enumerate(zip(predicted_tag_scores, true_tags_2d)):
        if len(pred_fields) != len(true_fields):
            raise ValueError(f"Sentence {sent_idx} has mismatched fields: {len(pred_fields)} predicted vs {len(true_fields)} gold")

        for field_idx_inner, (field_scores, true_field_tensor) in enumerate(zip(pred_fields, true_fields)):
            gold = true_field_tensor.view(-1)
            totals["D1"] += gold.numel()
            if gold.numel() == 0:
                continue

            preds = field_scores.argmax(dim=1)
            raw_preds = None
            if raw_unmasked_scores is not None and sent_idx < len(raw_unmasked_scores):
                raw_sent_fields = raw_unmasked_scores[sent_idx]
                if field_idx_inner < len(raw_sent_fields):
                    raw_field_scores = raw_sent_fields[field_idx_inner]
                    if raw_field_scores.numel() > 0:
                        raw_preds = raw_field_scores.argmax(dim=1)

            if can_compute and sent_idx < len(pos_list):
                pos_tags_for_sent = pos_list[sent_idx]
                for word_idx in range(gold.numel()):
                    gold_ix = gold[word_idx].item()
                    gold_tag_str = _gold_to_str(gold_ix, field_ix_to_tag)
                    pred_ix = preds[word_idx].item()
                    pred_tag_str = _gold_to_str(pred_ix, field_ix_to_tag)

                    if word_idx >= pos_tags_for_sent.numel():
                        continue

                    pos_ix = pos_tags_for_sent[word_idx].item()
                    pos_str = _pos_to_str(pos_ix, pos_ix_to_tag)
                    rule = lookup_rule_for_pos(pos_str, field_idx)

                    if is_rule_unconstrained(rule):
                        totals["unconstrained_tokens"] += 1
                        continue

                    totals["D2"] += 1
                    rule_set = rule_to_allowed_set(rule, field_idx)

                    if MASK_DEBUG_CHECK and rule_set and pred_tag_str not in rule_set:
                        illegal_prediction_violations.append({
                            "sent": sent_idx,
                            "word": word_idx,
                            "pos": pos_str,
                            "field": field_idx,
                            "pred": pred_tag_str,
                            "rule_samples": list(sorted(rule_set))[:5],
                        })

                    if gold_tag_str in {"_", None}:
                        totals["null_gold"] += 1
                    elif gold_tag_str not in rule_set:
                        totals["gold_illegal"] += 1
                    else:
                        totals["D3"] += 1
                        if preds[word_idx].item() == gold_ix:
                            totals["correct"] += 1
                        if raw_preds is not None and word_idx < raw_preds.numel() and raw_preds[word_idx].item() == gold_ix:
                            totals["raw_correct_d3"] += 1

                    if rule is not None:
                        totals["masked_tokens"] += 1
            else:
                for word_idx in range(gold.numel()):
                    if preds[word_idx].item() == gold[word_idx].item():
                        totals["correct"] += 1
                    if raw_preds is not None and word_idx < raw_preds.numel() and raw_preds[word_idx].item() == gold[word_idx].item():
                        totals["raw_correct_d3"] += 1
                totals["D3"] += gold.numel()

    filtered_accuracy = (totals["correct"] / totals["D3"]) if totals["D3"] > 0 else 0.0
    raw_at_d3 = (totals["raw_correct_d3"] / totals["D3"]) if totals["D3"] > 0 else None
    stats = {
        'D1': totals["D1"],
        'D2': totals["D2"],
        'D3': totals["D3"],
        'correct': totals["correct"],
        'raw_correct_d3': totals["raw_correct_d3"],
        'raw_at_d3': raw_at_d3,
        'masked_tokens': totals["masked_tokens"],
        'unconstrained_tokens': totals["unconstrained_tokens"],
        'null_gold': totals["null_gold"],
        'gold_illegal': totals["gold_illegal"],
        'filtered_accuracy': filtered_accuracy,
    }

    try:
        from debug_run import log_filtered_accuracy_stats
        log_filtered_accuracy_stats(0, totals["D1"], totals["D3"], totals["correct"])
    except Exception:
        pass

    if MASK_DEBUG_CHECK and illegal_prediction_violations:
        sample = illegal_prediction_violations[0]
        raise AssertionError(
            "[mask check] Illegal constrained prediction detected: "
            f"field={sample['field']} pos={sample['pos']} "
            f"pred={sample['pred']} allowed~{sample['rule_samples']} "
            f"sent={sample['sent']} word={sample['word']}"
        )

    return filtered_accuracy, stats


def log_mask_diagnostics(train_diag, val_diag, train_filt_stats, val_filt_stats, field_idx, val_sents):
    """
    Print statements related to masking:
    - [diag train] and [diag val]
    - [mask] train_masked_tokens and val_masked_tokens
    - [train filtered] and [val filtered] with D1/D2/D3
    - [filtering] message
    - [ENC unconstrained] message
    """
    from debug_run import diag_summary
    
    def warn_if_filtered_lt_raw(stats, split_label):
        if not stats or stats['D3'] == 0:
            return
        raw_at_d3_val = stats.get('raw_at_d3')
        if raw_at_d3_val is None:
            return
        filtered = stats['filtered_accuracy']
        # Allow a tiny tolerance for floating point noise
        if filtered + 1e-6 < raw_at_d3_val:
            print(f"[warn {split_label}] filtered({filtered:.4f}) < raw@D3({raw_at_d3_val:.4f}) "
                  f"on D3={stats['D3']}")

    if field_idx > 0:
        if train_filt_stats is not None:
            warn_if_filtered_lt_raw(train_filt_stats, "train")
        if val_sents and val_filt_stats is not None:
            warn_if_filtered_lt_raw(val_filt_stats, "val")

        if not VERBOSE_MASK_DEBUG:
            return

        print(f"[diag train] {diag_summary(train_diag)}")
        if val_sents and val_diag is not None:
            print(f"[diag  val ] {diag_summary(val_diag)}")
            print(
                f"[mask] train_masked_tokens={train_diag['masked_tokens']} val_masked_tokens={val_diag['masked_tokens']}")
        
        # Enhanced logging with D1/D2/D3 and filtering-strategy details
        if train_filt_stats is not None:
            raw_at_d3_val = train_filt_stats.get('raw_at_d3')
            raw_at_d3_str = f"{raw_at_d3_val:.4f}" if raw_at_d3_val is not None else "n/a (D3=0)" if train_filt_stats['D3'] == 0 else "n/a"
            print(f"[train filtered] Exclude filtering strategy (D3): "
                  f"D1={train_filt_stats['D1']} D2={train_filt_stats['D2']} D3={train_filt_stats['D3']} "
                  f"raw@D3={raw_at_d3_str} filtered={train_filt_stats['filtered_accuracy']:.4f} | "
                  f"unconstrained={train_filt_stats['unconstrained_tokens']} "
                  f"null_gold={train_filt_stats['null_gold']} "
                  f"gold_illegal={train_filt_stats['gold_illegal']} "
                  f"masked={train_filt_stats['masked_tokens']}")
        
        if val_sents and val_filt_stats is not None:
            raw_at_d3_val = val_filt_stats.get('raw_at_d3')
            raw_at_d3_str = f"{raw_at_d3_val:.4f}" if raw_at_d3_val is not None else "n/a (D3=0)" if val_filt_stats['D3'] == 0 else "n/a"
            print(f"[val filtered] Exclude filtering strategy (D3): "
                  f"D1={val_filt_stats['D1']} D2={val_filt_stats['D2']} D3={val_filt_stats['D3']} "
                  f"raw@D3={raw_at_d3_str} filtered={val_filt_stats['filtered_accuracy']:.4f} | "
                  f"unconstrained={val_filt_stats['unconstrained_tokens']} "
                  f"null_gold={val_filt_stats['null_gold']} "
                  f"gold_illegal={val_filt_stats['gold_illegal']} "
                  f"masked={val_filt_stats['masked_tokens']}")
            
            # Special message for ENC when unconstrained
            if field_idx == 4 and val_filt_stats['D2'] == 0:
                print(f"[ENC unconstrained] D2=0, masked_tokens=0, filtered=n/a")
            
            print(f"[filtering] Filtered accuracy computed with the Exclude strategy over D3 "
                  f"(constrained & non-null & non-illegal-gold tokens only). "
                  f"Fair comparison: filtered ≥ raw@D3")


def sanity_check_mask_activation(model, predict_fn, val_sents, val_pos_flat,
                                 field_tag_to_ix, field_ix_to_tag, pos_ix_to_tag,
                                 field_idx, field_name, legal_rules_for_field):
    """Confirm the mask is active by forcing a small forward pass."""
    if not val_sents:
        return

    model.eval()
    with torch.no_grad():
        tags_sample = predict_fn(model, val_sents[:2])

    masked_sample = apply_legal_mask(
        tags_sample,
        val_pos_flat[:2],
        field_tag_to_ix,
        field_ix_to_tag,
        pos_ix_to_tag,
        field_idx,
        field_name,
    )

    def _pick_head(heads):
        if len(heads) == 1:
            return heads[0]
        return heads[field_idx] if 0 <= field_idx < len(heads) else heads[0]

    any_flip = False
    for raw, msk in zip(tags_sample, masked_sample):
        raw_tensor = _pick_head(raw)
        msk_tensor = _pick_head(msk)

        if (msk_tensor == NEG).any():
            any_flip = True
            break

        if raw_tensor.argmax(-1).ne(msk_tensor.argmax(-1)).any():
            any_flip = True
            break

    expect_mask = expect_mask_for_field(
        field_idx=field_idx,
        legal_rules_for_field=legal_rules_for_field,
        field_tag_to_ix=field_tag_to_ix,
    )

    if expect_mask:
        assert any_flip, (
            f"[{field_name}] Mask expected but inactive — check call order / allowed_ix computation"
        )
    else:
        print(f"[sanity-check] masking skipped for '{field_name}' (no constraints for this field).")

    model.train()


def apply_training_mask_and_loss(
        logits,
        gold,
        field_idx,
        field_name,
        field_tag_to_ix,
        field_ix_to_tag,
        pos_ix_to_tag,
        pos_indices_flat,
        train_diag,
        loss_fn,
        treat_gold_illegal_as_underscore,
        legal_values,
):
    """Apply the legality mask during training and compute loss."""
    blank_ix = field_tag_to_ix.get('_', None)

    def _map_allowed(rule, gold_tag_str=None):
        if is_rule_unconstrained(rule):
            return None
        rset = set(rule) if isinstance(rule, (list, tuple, set)) else {rule}

        if analysis_key == "enclitic":
            if "NA" in field_tag_to_ix:
                rset.add("NA")
            if "_" in field_tag_to_ix:
                rset.add("_")

        mapped = [field_tag_to_ix[t] for t in rset if t in field_tag_to_ix]
        return torch.tensor(mapped, device=logits[0].device, dtype=torch.long) if mapped else None

    analysis_key = analysis_key_for_field(field_idx)

    legal_rules_for_field = get_legal_rules_for_field(field_idx)

    head_idx = field_idx if (
        isinstance(logits, (list, tuple)) and len(logits) > 1 and 0 <= field_idx < len(logits)
    ) else 0
    base = logits[head_idx]
    V = base.size(-1)
    view = base.reshape(-1, V).clone()

    pos_indices_flat = pos_indices_flat.reshape(-1)
    assert pos_indices_flat.numel() == view.size(
        0), f"POS/token mismatch: {pos_indices_flat.numel()} vs {view.size(0)}"

    for i in range(view.size(0)):
        pos_idx = int(pos_indices_flat[i])

        if isinstance(pos_ix_to_tag, dict):
            pos_str = pos_ix_to_tag.get(pos_idx, f"POS_{pos_idx}")
        elif isinstance(pos_ix_to_tag, (list, tuple)) and pos_idx < len(pos_ix_to_tag):
            pos_str = pos_ix_to_tag[pos_idx]
        else:
            pos_str = f"POS_{pos_idx}"

        gold_ix = gold[0].reshape(-1)[i].item()
        gold_tag_str = field_ix_to_tag.get(gold_ix, f"UNK_{gold_ix}")

        if gold_tag_str in {None, "_"}:
            train_diag["filtered_tokens"] = train_diag.get("filtered_tokens", 0) + 1
            continue

        pos_rules = legal_rules_for_field.get(pos_str) if field_idx < 4 else (
            legal_rules_for_field if isinstance(legal_rules_for_field, dict) else None
        )
        rule = pos_rules.get(analysis_key) if (pos_rules and analysis_key) else None

        if rule is not None:
            allowed_set = set(rule) if isinstance(rule, (list, tuple, set)) else {rule}
            if gold_tag_str not in allowed_set:
                train_diag["gold_illegal"] += 1
                continue

        allowed_ix = _map_allowed(rule, gold_tag_str=gold_tag_str)

        if allowed_ix is None or allowed_ix.numel() == 0:
            rule, decision = get_mask_and_loss_decision(
                field_idx, pos_str, gold_tag_str, legal_values,
                treat_gold_illegal_as_underscore
            )

            if decision == "skip_loss" and treat_gold_illegal_as_underscore and blank_ix is not None:
                gold[0].reshape(-1)[i] = blank_ix
                train_diag["gold_illegal"] += 1

            train_diag["pos_na"] += 1
            continue

        disallow = torch.ones(V, dtype=torch.bool, device=view.device)
        disallow[allowed_ix] = False
        if disallow.any():
            view[i, disallow] = NEG
            train_diag["masked_tokens"] += 1
            train_diag["saw_minus_inf"] = 1

        if treat_gold_illegal_as_underscore:
            gold_ix = gold[0].reshape(-1)[i].item()
            if gold_ix < len(field_ix_to_tag):
                gold_tag_str = field_ix_to_tag[gold_ix] if isinstance(field_ix_to_tag, dict) else field_ix_to_tag[gold_ix]
                if rule is not None and rule != "NA" and not (isinstance(rule, list) and rule == ["NA"]):
                    rule_set = set(rule) if isinstance(rule, (list, tuple, set)) else {rule}
                    if gold_tag_str not in rule_set:
                        if blank_ix is not None:
                            gold[0].reshape(-1)[i] = blank_ix
                        train_diag["gold_illegal"] += 1

    masked_head = view.view_as(base)
    logits_for_loss = list(logits) if isinstance(logits, (list, tuple)) else [logits]
    logits_for_loss[head_idx] = masked_head

    loss_vec = [loss_fn(l, g) for l, g in zip(logits_for_loss, gold)]
    return torch.stack(loss_vec).mean()


def apply_validation_mask_metrics(
        val_logits,
        val_tags,
        val_pos_flat,
        field_idx,
        field_name,
        field_tag_to_ix,
        field_ix_to_tag,
        pos_ix_to_tag,
        logits_dim,
        val_diag,
        debug_mask=False,
):
    """Mask validation logits, update diagnostics, and return filtered accuracy."""
    normalized_val_logits = normalize_logits_to_list_list(val_logits)
    
    for sent_idx, sent_logits in enumerate(normalized_val_logits):
        for word_idx in range(sent_logits[0].shape[0]):
            pos_gold_word = val_pos_flat[sent_idx][word_idx].item()
            pos_gold_str = _pos_to_str(pos_gold_word, pos_ix_to_tag)
            if pos_gold_str.startswith("POS_"):
                continue

            legal_rules_for_field = get_legal_rules_for_field(field_idx)
            if field_idx < 4:
                legal_rules_for_field = legal_rules_for_field.get(pos_gold_str, {})
                analysis_key = analysis_key_for_field(field_idx)
                rule = legal_rules_for_field.get(analysis_key, None)
            else:
                rule = legal_rules_for_field

            if rule and not is_rule_unconstrained(rule):
                allowed_tags = {t for t in rule if t != "_" and t in field_tag_to_ix}
                if not allowed_tags and "_" in field_tag_to_ix:
                    allowed_tags = {"_"}
                allowed_ix = {field_tag_to_ix[t] for t in allowed_tags if t in field_tag_to_ix}
                if len(allowed_ix) < logits_dim:
                    val_diag["masked_tokens"] += 1

    val_masked = apply_legal_mask(
        normalized_val_logits,
        val_pos_flat,
        field_tag_to_ix,
        field_ix_to_tag,
        pos_ix_to_tag,
        field_idx,
        field_name,
    )

    for sent_idx, sent_masked in enumerate(val_masked):
        field_masked = sent_masked[0]
        for word_idx in range(field_masked.shape[0]):
            val_diag["total_tokens"] += 1
            pred_filt_ix = field_masked[word_idx].argmax().item()
            if pred_filt_ix < len(field_ix_to_tag):
                pred_tag = field_ix_to_tag.get(pred_filt_ix, "") if isinstance(field_ix_to_tag, dict) else field_ix_to_tag[pred_filt_ix]
                if pred_tag == "_":
                    val_diag["pred_uscore"] += 1
            else:
                print(
                    f"Warning: pred_filt_ix {pred_filt_ix} out of range for field {field_idx} "
                    f"(vocab size: {len(field_ix_to_tag)})"
                )

            gold_ix = val_tags[sent_idx][0][word_idx].item()
            if gold_ix < len(field_ix_to_tag):
                gold_tag_str = field_ix_to_tag.get(gold_ix, f"TAG_{gold_ix}") if isinstance(field_ix_to_tag, dict) else field_ix_to_tag[gold_ix]
            else:
                print(
                    f"Warning: gold_ix {gold_ix} out of range for field {field_idx} "
                    f"(vocab size: {len(field_ix_to_tag)})"
                )
                continue

            pos_gold_word = val_pos_flat[sent_idx][word_idx].item()
            if isinstance(pos_ix_to_tag, dict):
                pos_gold_str = pos_ix_to_tag.get(pos_gold_word, f"POS_{pos_gold_word}")
            elif isinstance(pos_ix_to_tag, (list, tuple)) and pos_gold_word < len(pos_ix_to_tag):
                pos_gold_str = pos_ix_to_tag[pos_gold_word]
            else:
                print(
                    f"Warning: pos_gold_word {pos_gold_word} out of range for POS "
                    f"(vocab size: {len(pos_ix_to_tag)})"
                )
                continue

            legal_rules_for_field = get_legal_rules_for_field(field_idx)
            if is_gold_illegal_or_none(gold_tag_str, pos_gold_str, field_idx, legal_rules_for_field):
                val_diag["gold_illegal"] += 1

    if debug_mask:
        debug_len_pairs(val_masked, val_tags, ctx="chk/val_filt")
        assert_TV_shapes(val_masked, val_tags, ctx="val_filt")

    legal_rules_for_field = get_legal_rules_for_field(field_idx)
    val_filt_acc, val_filt_stats = calculate_accuracy_for_filtered_predictions(
        val_masked,
        val_tags,
        pos_list=val_pos_flat,
        field_idx=field_idx,
        field_tag_to_ix=field_tag_to_ix,
        field_ix_to_tag=field_ix_to_tag,
        pos_ix_to_tag=pos_ix_to_tag,
        legal_rules_for_field=legal_rules_for_field,
        raw_unmasked_scores=normalized_val_logits,
    )

    return val_filt_acc, val_filt_stats


def count_illegal_tags_for_field(training_data, legal_values, field_idx, field_name):
    """
    Count illegal tags for a single field in the training data.
    
    """
    field_stats = {
        "D1": 0,
        "D2": 0,
        "D3": 0,
        "illegal": 0,
        "unconstrained_tokens": 0,
        "null_gold": 0,
        "illegal_distribution": Counter()
    }
    
    for sent_idx, (sent, tag_sets) in enumerate(training_data):
        for word_idx, tag_set in enumerate(tag_sets):
            if len(tag_set) == 0:
                continue
            
            # POS is the first element
            pos_value = tag_set[0]
            
            # get the tag value for this field
            if field_idx < len(tag_set):
                tag_value = tag_set[field_idx]
                
                # skip if tag is empty or None
                if tag_value is None or tag_value == "":
                    continue
                
                # D1: all tokens
                field_stats["D1"] += 1
                
                rule = lookup_rule(field_idx, pos_value, legal_values)
                is_unconstrained = is_rule_unconstrained(rule)
                
                if is_unconstrained:
                    # unconstrained tokens excluded from D2 and D3
                    field_stats["unconstrained_tokens"] += 1
                else:
                    # constrained token - counts toward D2
                    field_stats["D2"] += 1
                    
                    # check if Null gold
                    if tag_value in {"_", None}:
                        field_stats["null_gold"] += 1
                        # null gold excluded from D3
                    else:
                        # check if illegal gold
                        if isinstance(rule, (list, tuple, set)):
                            rule_set = set(rule)
                        else:
                            rule_set = {rule}
                        
                        if tag_value not in rule_set:
                            # illegal gold - excluded from D3, but counted
                            field_stats["illegal"] += 1
                            field_stats["illegal_distribution"][tag_value] += 1
                        else:
                            # legal, non-null gold - counts toward D3
                            field_stats["D3"] += 1
    
    return field_stats


def write_field_statistics_to_file(field_stats, field_name, output_file_path, is_first_field=False, logger=None):
    """Append formatted illegal-tag stats for a field to disk."""
    output_lines = []
    
    # Write header only for the first field
    if is_first_field:
        output_lines.append("="*80)
        output_lines.append("ILLEGAL TAG STATISTICS (TRAINING DATA)")
        output_lines.append("Using Exclude filtering strategy: illegal / D3 (constrained & non-null & non-illegal-gold)")
        output_lines.append("="*80)
    
    D1 = field_stats.get("D1", 0)
    D2 = field_stats.get("D2", 0)
    D3 = field_stats.get("D3", 0)
    illegal = field_stats["illegal"]
    unconstrained = field_stats.get("unconstrained_tokens", 0)
    null_gold = field_stats.get("null_gold", 0)
    
    # Primary illegality metric: illegal / D3
    illegal_pct_d3 = (illegal / D3 * 100) if D3 > 0 else 0.0
    
    # Additional transparency metrics
    unconstrained_pct_d1 = (unconstrained / D1 * 100) if D1 > 0 else 0.0
    null_gold_pct_d2 = (null_gold / D2 * 100) if D2 > 0 else 0.0
    
    output_lines.append(f"\nField: {field_name.upper()}")
    output_lines.append(f"  Denominators:")
    output_lines.append(f"    D1 (all tokens): {D1}")
    output_lines.append(f"    D2 (constrained tokens): {D2}")
    output_lines.append(f"    D3 (D2 excluding Null gold & illegal-gold): {D3}")
    output_lines.append(f"  Primary illegality metric: {illegal} / {D3} = {illegal_pct_d3:.2f}%")
    output_lines.append(f"  Exclusions:")
    output_lines.append(f"    Unconstrained tokens: {unconstrained} / {D1} = {unconstrained_pct_d1:.2f}%")
    output_lines.append(f"    Null gold (\"_\"): {null_gold} / {D2} = {null_gold_pct_d2:.2f}%")
    
    if illegal > 0:
        output_lines.append(f"  Illegal tag distribution:")
        sorted_dist = sorted(field_stats["illegal_distribution"].items(), 
                           key=lambda x: x[1], reverse=True)
        for tag_value, count in sorted_dist:
            pct = (count / illegal * 100) if illegal > 0 else 0
            output_lines.append(f"    {tag_value}: {count} ({pct:.2f}%)")
    else:
        output_lines.append(f"  No illegal tags found")
    
    output_text = "\n".join(output_lines)
    
    # Write to file
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        mode = 'w' if is_first_field else 'a'
        with open(output_file_path, mode, encoding='utf-8') as f:
            f.write(output_text)
        if is_first_field and logger:
            logger.info(f"Illegal tag statistics file created: {output_file_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Could not write illegal tag statistics to file: {e}")
        print(output_text)  # Fallback to stdout


def collect_and_write_illegal_tag_statistics(training_data, field_names, model_path, logger=None):
    """
    Collect illegal tag statistics for all non-POS fields and write to file.
    """
    try:
        legal_values = load_legal_values()
        if not legal_values:
            if logger:
                logger.debug("legal_values.json not found, skipping illegal tag statistics")
            return False
        
        model_dir = os.path.dirname(model_path)
        stats_file_path = os.path.join(model_dir, "illegal_tags_statistics.txt")
        
        for field_idx, field_name in enumerate(field_names[1:], start=1):
            try:

                field_stats = count_illegal_tags_for_field(training_data, legal_values, field_idx, field_name)

                is_first_field = (field_idx == 1)
                write_field_statistics_to_file(field_stats, field_name, stats_file_path, 
                                              is_first_field=is_first_field, logger=logger)
            except Exception as e:
                if logger:
                    logger.warning(f"Could not analyze illegal tags for {field_name}: {e}")
        
        try:
            with open(stats_file_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
        except Exception:
            pass
        
        return True
        
    except Exception as e:
        if logger:
            logger.warning(f"Could not collect illegal tag statistics: {e}")
        return False


def expect_mask_for_field(field_idx, legal_rules_for_field, field_tag_to_ix):
    if not legal_rules_for_field:
        return False
    
    try:
        from run import is_rule_unconstrained
    except ImportError:
        def is_rule_unconstrained(rule):
            if rule is None:
                return True
            if rule == "NA":
                return True
            if isinstance(rule, list) and (len(rule) == 0 or rule == ["NA"]):
                return True
            if isinstance(rule, (set, tuple)) and len(rule) == 0:
                return True
            return False
    
    # Handle ENC (field_idx == 4) as flat list
    if field_idx == 4:
        # ENC: legal_rules_for_field is the flat list directly
        rule = legal_rules_for_field
        
        # Structural guard: warn if ENC is a dict
        if isinstance(rule, dict):
            print(f"[expect_mask field=4 (ENC)] ERROR: enclitic rules should be a list, not dict. Got: {type(rule)}")
            return False
        
        # Use single source of truth for unconstrained detection
        if is_rule_unconstrained(rule):
            return False
        
        # ENC has constraints: check if it restricts vocabulary
        allowed = set(rule) if isinstance(rule, (list, tuple, set)) else {rule}
        if "_" not in allowed:
            allowed.add("_")
        if len(allowed) < len(field_tag_to_ix):
            return True  # Masking expected
        return False
    else:
        # an1/an2/an3: POS-conditioned rules (dict structure)
        if not isinstance(legal_rules_for_field, dict):
            return False
            
        for pos_label, rules in legal_rules_for_field.items():
            rule = rules.get(f"analysis{field_idx}")
            
            if is_rule_unconstrained(rule):
                continue
                
            allowed = set(rule) if isinstance(rule, (list, tuple, set)) else {rule}
            if "_" not in allowed:
                allowed.add("_")
            if len(allowed) < len(field_tag_to_ix):
                return True
        return False


def scan_field_rules_and_check_constraints(field_idx, field_name, val_poses, ix_to_pos, ix_to_field_tag, field_tag_to_ix):
    """Run the rule scan + expect_mask checks for a field."""
    
    # Skip for POS
    if field_idx <= 0:
        return  
    
    legal_rules_for_field = get_legal_rules_for_field(field_idx)
    
    rulescan_for_field(field_idx, val_poses, [ix_to_pos, ix_to_field_tag], legal_rules_for_field)
    
    # check if field has effective constraints
    expect_mask = expect_mask_for_field(field_idx, legal_rules_for_field, field_tag_to_ix)
    if not expect_mask:
        print(f"[rules] {field_name}: no effective constraints found; mask will be a no-op.")

