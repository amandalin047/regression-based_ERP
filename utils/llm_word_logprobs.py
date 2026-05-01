from transformers import BertTokenizerFast, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd

SPECIAL_TOKENS = "[PAD]", "[CLS]", "[SEP]"
SPECIAL_TOKEN_IDS = 0, 101, 102 # BERT token IDs: 0 = [PAD]; 101 = [CLS]; 102 = [SEP]
OTHER_TOKENS = "[UNK]", "##"
MAX_LENGTH = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sentences_and_splits(file_dir: str, file_name: str, sheet_name: str | None, *,
                             frame_col: str, word_cols: list, 
                             group_by: str | None=None, 
                             prompt: str | None=None) -> tuple | list:
    
    def clean_punc(cell):
        if isinstance(cell, str):
            to_replace = {",": "，",
                      ".": "。",
                      " ，": "，",
                      "， ": "，",
                      " 。": "。",
                      "。 ": "。"}
            for k, v in to_replace.items():
                cell = cell.replace(k, v)
        return cell
    
    def for_each_group(df_per_group):
        frames = df_per_group[frame_col].to_list()

        if prompt is None:
            print(f"Prompts is not set. Be sure that the stimuli presentation did not include a judgement task with a prompt lie `通順嗎？`")
            split_sentences = [df_per_group.iloc[i][word_cols].dropna().to_list()
                                                                for i in range(len(df_per_group))]
    
        else:
            last_words = split_sentences = [df_per_group.iloc[i][word_cols].dropna().to_list()[-1]
                                                                for i in range(len(df_per_group))]
            if last_words.count(prompt) != len(last_words):
                raise ValueError(f"There's at least one sentence whose final word is not {prompt}!")
            split_sentences = [df_per_group.iloc[i][word_cols].dropna().to_list()[:-1]
                                                                for i in range(len(df_per_group))]

        sentences = [f + s[-1] for f, s in zip(frames, split_sentences)]
        sentences_stitched = ["".join(s) for s in split_sentences]
        assert sentences_stitched == sentences, "Splitted sentnce and original sentence frames mistmatch! Please check for character or punctuation mismatch!"

        return sentences, split_sentences
    
    cols =  [frame_col] + word_cols if group_by is None else [frame_col] + word_cols + [group_by]
    df = pd.read_excel(f"{file_dir}/{file_name}", sheet_name=sheet_name)[cols]
    df = df.map(clean_punc)
    df = df.map(lambda cell: cell.strip() if isinstance(cell, str) else cell)

    if group_by is None:
        return for_each_group(df)
    else:
        groups = df[group_by].unique()
        df_per_group_list = [df.loc[df[group_by] == i] for i in groups]
        sent_split_sent_list = [for_each_group(df_per_group) for df_per_group in df_per_group_list]
        return sent_split_sent_list


def get_word_logprobs(*, tokens: list, split_sentences: list,
                      ids: np.ndarray, token_logprobs: np.ndarray,
                      special_tokens: tuple | list | None=None,
                      special_token_ids: tuple| list| None=None,
                      other_tokens: tuple | list | None=None) -> list:
    if special_tokens is None or special_token_ids is None:   
        print(f"Special tokens or special token IDs not set.\nDefaulting to 0 = '[PAD]'; 101 = '[CLS]'; 102 = '[SEP]'.")
        special_tokens = SPECIAL_TOKENS
        special_token_ids = SPECIAL_TOKEN_IDS
    if other_tokens is None:
        print(f"Other tokens not set.\nDefaulting to {OTHER_TOKENS}.")
        other_tokens = OTHER_TOKENS

    tokens_no_special = [list(filter(lambda s: s not in special_tokens, t)) for t in tokens]
    
    spans = [[0] for _ in tokens_no_special]
    for i, (trow, srow) in enumerate(zip(tokens_no_special, split_sentences)):
        sucess = True
        chunk = ""
        idx = 0
        for j, t in enumerate(trow):
            if t in other_tokens or t.startswith("#"):
                print(f"{t} found in sentence index {i}, token index {j}:\n  {trow} \n  {srow}")
                sucess = False
                break
            if idx >= len(srow):
                print(f"Index out of range in sentence index {i}, token index {j}; idx = {idx}:\n  {trow} \n  {srow}")
                sucess = False
                break
            if srow[idx].startswith(chunk + t):
                chunk += t
            else:
                chunk = t
                idx += 1
                spans[i].append(j)
        if sucess:
            spans[i].append(len(tokens_no_special[i]))
    # print(spans) <- for debugging

    check_span = [["".join(trow[seq[i]: seq[i+1]]) for i in range(len(seq)-1)]
                                          for trow, seq in zip(tokens_no_special, spans)]
    # print(check_span) <- for debugging
    aligned = np.array([check == split for check, split in zip(check_span, split_sentences)])
    not_aligned_where = np.where(aligned == False)[0]
    if not_aligned_where.size != 0:
        print(f">> Sentence indices where alignment failed: {not_aligned_where}\n")
    
    token_logprobs_masked = [tprobs[~ np.isin(_id, special_token_ids)] for _id, tprobs in zip(ids, token_logprobs)]
    word_logprobs = [[sum(tprobs[seq[i]: seq[i+1]]) for i in range(len(seq)-1)]
                                                    for tprobs, seq in zip(token_logprobs_masked, spans)]
    return word_logprobs

def get_llm_token_logprobs(sentences: list,
                           model: AutoModelForCausalLM, tokenizer: BertTokenizerFast,
                           max_length: int=MAX_LENGTH) -> tuple:
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    elif tokenizer.pad_token_id is None:
        raise ValueError ("Tokenizer pad token ID is None. Cannot safely use padding=True without defining a pad token.")
    
    if model.config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("Model and tokenizer pad token ID mismatch!")
    
    inputs = tokenizer(sentences,
                   truncation=True,
                   padding=True,
                   return_tensors="pt",
                   return_offsets_mapping=False,
                   max_length=max_length)
    assert "attention_mask" in inputs, "Tokenizer output (input to model) has no attention_mask!"
    assert inputs["attention_mask"].shape == inputs["input_ids"].shape

    inputs = {k: v.to(device) for k, v in inputs.items()}
    ids = inputs["input_ids"]     
    ids_cpu = ids.detach().cpu().numpy()  # tokenizer usually expects integer IDs, not tensors 
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in ids_cpu]
    
    model.to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model(**inputs,
                        output_hidden_states=False)
    logits = outputs.logits    # shape = (batch_size, seq_len, vocab_size)
    logprobs = torch.log_softmax(logits, dim=-1)

    target_ids = ids[:, 1:]
    token_logprobs = torch.gather(logprobs[:, :-1, :],
                                  dim=-1,
                                  index=target_ids.unsqueeze(-1)).squeeze(-1)
    token_logprobs = torch.cat([torch.zeros((target_ids.size(0), 1), device=token_logprobs.device),
                                token_logprobs],
                                dim=1)
    token_logprobs = token_logprobs.detach().cpu().numpy()
    # token_logprobs = np.array([[logprobs[i, j, idx] for j, idx in enumerate(target_ids[i])]
                                       # for i in range(target_ids.size(0))]) <- works, but too many for loops
    
    return ids_cpu, tokens, token_logprobs


if __name__ == "__main__":
    file_dir = "/Users/jowanglin/regression-based_ERP/data/stimuli"
    file_name = "CRYSTAL_master-sheet.xlsx"
    sheet_name = "Overall"
    frame_col = "Sentence_frame"
    word_cols = [f"W{i}" for i in range(1,13)]
    prompt = "通順嗎？"

    sentences, split_sentences = get_sentences_and_splits(file_dir,
                                                          file_name,
                                                          sheet_name,
                                                          frame_col=frame_col,
                                                          word_cols=word_cols,
                                                          prompt=prompt)
    print(sentences[:3])
    print(split_sentences[:3])
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    model = AutoModelForCausalLM.from_pretrained("ckiplab/gpt2-base-chinese") 
    
    ids, tokens, token_logprobs = get_llm_token_logprobs(sentences,
                                                         model, 
                                                         tokenizer,
                                                         max_length=MAX_LENGTH)
    
    pad_token, cls_token, sep_token = tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token
    pad_id, cls_id, sep_id = tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id
    
    word_logprobs = get_word_logprobs(tokens=tokens,
                                      split_sentences=split_sentences,
                                      ids=ids,
                                      token_logprobs=token_logprobs,
                                      special_tokens=(pad_token, cls_token, sep_token),
                                      special_token_ids=(pad_id, cls_id, sep_id),
                                      other_tokens=OTHER_TOKENS)