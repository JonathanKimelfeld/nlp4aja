from collections import defaultdict

DEBUG_MASK = False


def new_diag():
    return {
        "pos_na": 0,
        "gold_illegal": 0,
        "mask_empty_fallback": 0,
        "masked_tokens": 0,
        "pred_uscore": 0,
        "total_tokens": 0,
        "saw_minus_inf": 0,
    }

def diag_summary(d):
    denom = max(1, d.get("total_tokens", 0))
    pct_u = 100.0 * d.get("pred_uscore", 0) / denom
    return (f"pos_na={d.get('pos_na', 0)} gold_illegal={d.get('gold_illegal', 0)} "
            f"mask_empty_fallback={d.get('mask_empty_fallback', 0)} "
            f"masked_tokens={d.get('masked_tokens', 0)} "
            f"pred_'_'={d.get('pred_uscore', 0)}/{denom} ({pct_u:.2f}%) "
            f"saw_-1e9={d.get('saw_minus_inf', 0) > 0}")


def log_train_tag_overview(field_idx, train_pos_list, train_tags, val_pos_list, val_tags):
    if not DEBUG_MASK:
        return
    try:
        print(f"[DEBUG] train_tag: field_idx={field_idx}")
        print(f"[DEBUG] train_pos_list sample: {train_pos_list[:2]}")
        print(f"[DEBUG] train_tags sample: {train_tags[:2]}")
        print(f"[DEBUG] val_pos_list sample: {val_pos_list[:2]}")
        print(f"[DEBUG] val_tags sample: {val_tags[:2]}")
    except Exception:
        pass


def log_filtered_accuracy_stats(total_filtered_out, total_raw_tokens, total_words, total_correct):
    if not DEBUG_MASK:
        return
    try:
        filtering_percentage = (total_filtered_out / total_raw_tokens * 100) if total_raw_tokens > 0 else 0
        print(f"[DEBUG] Filtered {total_filtered_out}/{total_raw_tokens} tokens ({filtering_percentage:.1f}%)")
        print(f"[DEBUG] Accuracy computed on {total_words} tokens, {total_correct} correct")
    except Exception:
        pass


def debug_len_pairs(pred_sents, gold_sents, ctx="chk", max_show=6):
    if not DEBUG_MASK:
        return
    try:
        for i, (pred_fields, gold_fields) in enumerate(zip(pred_sents, gold_sents)):
            if not pred_fields or not gold_fields:
                print(f"[{ctx}] sent{i}: (empty fields)")
                continue
            pred = pred_fields[0]
            gold = gold_fields[0]
            pT = pred.shape[0] if hasattr(pred, "shape") and pred.dim() >= 1 else len(pred)
            gT = gold.numel()
            print(f"[{ctx}] sent{i}: pred_T={pT} gold_T={gT}")
            if i + 1 >= max_show:
                break
        for i, (pred_fields, gold_fields) in enumerate(zip(pred_sents, gold_sents)):
            if not pred_fields or not gold_fields:
                continue
            pred = pred_fields[0]
            gold = gold_fields[0]
            pT = pred.shape[0] if hasattr(pred, "shape") and pred.dim() >= 1 else len(pred)
            gT = gold.numel()
            if pT != gT:
                print(f"[{ctx}] MISMATCH at sent{i}: pred_T={pT} gold_T={gT}")
                break
        print(f"[{ctx}] OK: all checked [T] match gold lengths.")
    except Exception as e:
        print(f"[{ctx}] DEBUG ERROR: {e}")


def assert_TV_shapes(pred_sents, gold_sents, ctx="acc"):
    if not DEBUG_MASK:
        return
    try:
        for i, (pred_fields, gold_fields) in enumerate(zip(pred_sents, gold_sents)):
            if not pred_fields or not gold_fields:
                continue
            pred = pred_fields[0]
            gold = gold_fields[0]
            assert hasattr(pred, "dim"), f"[{ctx}] pred tensor has no dim() at sent {i}"
            assert pred.dim() == 2 and pred.size(1) > 1, \
                f"[{ctx}] pred tensor should be [T,V], got {list(pred.size())}"
            assert hasattr(gold, "dim"), f"[{ctx}] gold tensor has no dim() at sent {i}"
            assert gold.dim() == 1, f"[{ctx}] gold tensor should be [T], got {list(gold.size())}"
        print(f"[{ctx}] tensors look like [T,V] and [T]")
    except Exception as e:
        print(f"[{ctx}] DEBUG ERROR: {e}")


def ensure_2d_tensor(x):
    try:
        if hasattr(x, "dim"):
            return x if x.dim() > 1 else x.unsqueeze(0)
        if isinstance(x, list):
            if len(x) == 0:
                return torch.empty(0, 0)
            if hasattr(x[0], "dim"):
                dev = getattr(x[0], "device", None)
                x = torch.stack(x, dim=0)
                if dev is not None:
                    x = x.to(dev)
                return x if x.dim() > 1 else x.unsqueeze(0)
            x = torch.as_tensor(x, dtype=torch.float32)
            return x if x.dim() > 1 else x.unsqueeze(0)
        if np is not None and isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            return x if x.dim() > 1 else x.unsqueeze(0)
        x = torch.as_tensor(x)
        return x if x.dim() > 1 else x.unsqueeze(0)
    except Exception:
        return x


def assert_bijection(tag_to_ix, ix_to_tag, field_name):
    assert isinstance(tag_to_ix, dict), f"[{field_name}] tag_to_ix must be dict(tag->ix)"
    if isinstance(ix_to_tag, dict):
        n = len(ix_to_tag)
        assert set(ix_to_tag.keys()) == set(range(n)), \
            f"[{field_name}] ix_to_tag keys must be 0..{n - 1}"
        assert len(tag_to_ix) == n, \
            f"[{field_name}] size mismatch tag_to_ix={len(tag_to_ix)} ix_to_tag={n}"
        seen = set()
        for t, i in tag_to_ix.items():
            assert isinstance(t, str), f"[{field_name}] non-str tag in tag_to_ix: {t!r}"
            assert isinstance(i, int) and 0 <= i < n, f"[{field_name}] bad index for tag '{t}': {i}"
            assert i not in seen, f"[{field_name}] duplicate index {i} from tag '{t}'"
            seen.add(i)
            assert ix_to_tag[i] == t, f"[{field_name}] round-trip fails at {i}: '{t}' -> {ix_to_tag[i]!r}"
        for i, t in ix_to_tag.items():
            assert tag_to_ix.get(t, None) == i, \
                f"[{field_name}] reverse round-trip fails at {i}: '{t}' -> {tag_to_ix.get(t)}"
    elif isinstance(ix_to_tag, (list, tuple)):
        n = len(ix_to_tag)
        assert len(tag_to_ix) == n, \
            f"[{field_name}] size mismatch tag_to_ix={len(tag_to_ix)} ix_to_tag={n}"
        seen = set()
        for t, i in tag_to_ix.items():
            assert isinstance(t, str), f"[{field_name}] non-str tag in tag_to_ix: {t!r}"
            assert isinstance(i, int) and 0 <= i < n, f"[{field_name}] bad index for tag '{t}': {i}"
            assert i not in seen, f"[{field_name}] duplicate index {i} from tag '{t}'"
            seen.add(i)
            assert ix_to_tag[i] == t, f"[{field_name}] round-trip fails at {i}: '{t}' -> {ix_to_tag[i]!r}"
        for i, t in enumerate(ix_to_tag):
            assert tag_to_ix.get(t, None) == i, \
                f"[{field_name}] reverse round-trip fails at {i}: '{t}' -> {tag_to_ix.get(t)}"
    else:
        raise TypeError(f"[{field_name}] ix_to_tag must be dict or list/tuple")


def debug_validate_tag_inputs(field_idx, train_pos_list, train_sents, train_tags,
                              val_pos_list, val_sents, val_tags, tag_to_ix_list):
    """Debug-only sanity checks for tag tensors."""
    log_train_tag_overview(field_idx, train_pos_list, train_tags, val_pos_list, val_tags)

    assert len(train_pos_list) == len(train_sents) == len(train_tags), "Mismatch in training data lengths!"
    assert len(val_pos_list) == len(val_sents) == len(val_tags), "Mismatch in validation data lengths!"

    tag_dict = tag_to_ix_list[1] if len(tag_to_ix_list) > 1 else tag_to_ix_list[0]

    for tags in train_tags[:5]:
        for tag_tensor in tags:
            for tag in tag_tensor:
                assert tag.item() in tag_dict.values(), f"[ASSERT] Train tag {tag.item()} not in tag dict!"

    for tags in val_tags[:5]:
        for tag_tensor in tags:
            for tag in tag_tensor:
                assert tag.item() in tag_dict.values(), f"[ASSERT] Val tag {tag.item()} not in tag dict!"
