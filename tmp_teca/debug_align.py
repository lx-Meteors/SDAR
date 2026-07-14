import sys
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/home/test/yyy/SDAR")
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.workers.actor.dp_actor import DataParallelPPOActor

model_path = "/home/test/models/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

texts = ["The capital of France is Paris. The capital of Germany is",
         "1+1=2. 2+2=4. 3+3="]
resp_texts = [" Berlin, and the capital of Italy is Rome.", "6. 4+4=8."]

prompt_len, resp_len = 24, 16
input_ids, attn, resp = [], [], []
for t, r in zip(texts, resp_texts):
    p_ids = tok.encode(t)[:prompt_len]
    r_ids = tok.encode(r)[:resp_len]
    pad_p = prompt_len - len(p_ids)
    pad_r = resp_len - len(r_ids)
    ids = [tok.pad_token_id] * pad_p + p_ids + r_ids + [tok.pad_token_id] * pad_r
    m = [0] * pad_p + [1] * (len(p_ids) + len(r_ids)) + [0] * pad_r
    input_ids.append(ids); attn.append(m); resp.append(r_ids + [tok.pad_token_id] * pad_r)

input_ids = torch.tensor(input_ids, device="cuda")
attn = torch.tensor(attn, device="cuda")
resp = torch.tensor(resp, device="cuda")
pos = torch.clip(torch.cumsum(attn, dim=-1) - 1, min=0)
micro = {"input_ids": input_ids, "attention_mask": attn, "position_ids": pos, "responses": resp}

outs = {}
with torch.no_grad():
    for rmpad in [True, False]:
        cfg = OmegaConf.create({"use_remove_padding": rmpad, "use_fused_kernels": False,
                                "ulysses_sequence_parallel_size": 1, "use_torch_compile": False, "grad_clip": 1.0})
        actor = DataParallelPPOActor(config=cfg, actor_module=model)
        outs[rmpad] = actor._forward_micro_batch_candidates(micro, temperature=1.0, top_k=8)
        # also original verl log_prob path for reference
        ent, lp = actor._forward_micro_batch(micro, temperature=1.0, calculate_entropy=False)
        outs[(rmpad, "orig")] = lp

valid = attn[:, -resp_len:].bool()
a, b = outs[True]["realized_log_probs"], outs[False]["realized_log_probs"]
diff = (a - b).abs()
for i in range(2):
    print(f"sample {i}: valid={valid[i].int().tolist()}")
    print("  rmpad :", [round(x, 2) for x in a[i].tolist()])
    print("  dense :", [round(x, 2) for x in b[i].tolist()])
    print("  origT :", [round(x, 2) for x in outs[(True, 'orig')][i].tolist()])
    print("  origF :", [round(x, 2) for x in outs[(False, 'orig')][i].tolist()])
    print("  diff  :", [round(x, 2) for x in diff[i].tolist()])
