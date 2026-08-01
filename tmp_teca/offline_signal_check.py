"""
Offline signal check for Teacher-Entropy Credit Assignment (TECA).

For each generated response token position:
  1. teacher = same model, prompt prefixed with privileged skill info
     (identical construction to verl/trainer/ppo/rlsd_ray_trainer.build_teacher_batch)
  2. candidate set C_t = teacher top-k token ids
  3. renormalize teacher/student probs within C_t, compute entropies H_T, H_S
  4. delta_H = H_T - H_S
  5. eligibility: realized token is top-1 in BOTH teacher and student distributions

Reports: top-1 agreement rate, delta_H distribution on eligible tokens,
fraction of tokens that would receive a boost, example boosted tokens.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/test/models/Qwen2.5-3B-Instruct"
SKILLS_DIR = "/home/test/yyy/SDAR/skills/webshop"
TOP_K = 8
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 384
N_SAMPLES_PER_PROMPT = 4
OUT_PATH = os.path.join(os.path.dirname(__file__), "signal_stats.json")

sys.path.insert(0, "/home/test/yyy/SDAR")
from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS  # noqa: E402
from verl.trainer.ppo.rlsd_utils import SkillProvider  # noqa: E402


TASKS = [
    {
        "task": "i am looking for a pair of black running shoes size 10 for men, and price lower than 60 dollars",
        "obs": (
            "WebShop [SEP] Instruction: [SEP] i am looking for a pair of black running shoes size 10 for men, "
            "and price lower than 60 dollars [SEP] Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] "
            "B09Q3F6L2M [SEP] Mens Running Shoes Lightweight Athletic Walking Sneakers Black [SEP] $45.99 [SEP] "
            "B08XY45KLN [SEP] Men's Trail Running Shoes Non Slip Gym Sneakers Grey [SEP] $62.5 [SEP] "
            "B07TQZR8VP [SEP] Fashion Casual Canvas Shoes for Men White [SEP] $29.99 [SEP] "
            "B09KL2MNOP [SEP] Mens Black Running Shoes Breathable Mesh Tennis Sneakers [SEP] $55.0"
        ),
        "actions": '"click[b09q3f6l2m]", "click[b08xy45kln]", "click[b07tqzr8vp]", "click[b09kl2mnop]", "click[next >]", "search[...]"',
    },
    {
        "task": "find me a machine washable grey throw pillow cover set of 2, 18x18 inch, price lower than 25 dollars",
        "obs": (
            "WebShop [SEP] Instruction: [SEP] find me a machine washable grey throw pillow cover set of 2, 18x18 inch, "
            "price lower than 25 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] grey [SEP] beige [SEP] navy [SEP] "
            "size [SEP] 18x18 inch [SEP] 20x20 inch [SEP] Throw Pillow Covers Set of 2 Soft Velvet Decorative Cushion Cases [SEP] "
            "Price: $19.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        ),
        "actions": '"click[back to search]", "click[< prev]", "click[grey]", "click[beige]", "click[navy]", "click[18x18 inch]", "click[20x20 inch]", "click[description]", "click[features]", "click[reviews]", "click[buy now]"',
    },
    {
        "task": "i need a usb c fast charging cable 6ft 2 pack that is nylon braided, and price lower than 15 dollars",
        "obs": (
            "WebShop [SEP] Instruction: [SEP] i need a usb c fast charging cable 6ft 2 pack that is nylon braided, "
            "and price lower than 15 dollars [SEP] Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] "
            "B0912ABCDE [SEP] USB C Cable 6ft 2 Pack Nylon Braided Fast Charging Cord [SEP] $11.99 [SEP] "
            "B0834FGHIJ [SEP] USB Type C Charger Cable 3ft Short Braided [SEP] $8.99 [SEP] "
            "B0756KLMNO [SEP] Lightning Cable for iPhone 6ft 2 Pack [SEP] $13.99 [SEP] "
            "B0678PQRST [SEP] USB C to USB C Cable 100W 10ft Single Pack [SEP] $16.99"
        ),
        "actions": '"click[b0912abcde]", "click[b0834fghij]", "click[b0756klmno]", "click[b0678pqrst]", "click[next >]", "search[...]"',
    },
    {
        "task": "i want to buy a fragrance free moisturizing body lotion for sensitive skin, 20 ounce, price lower than 20 dollars",
        "obs": (
            "WebShop [SEP] Instruction: [SEP] i want to buy a fragrance free moisturizing body lotion for sensitive skin, "
            "20 ounce, price lower than 20 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 12 ounce [SEP] 20 ounce [SEP] "
            "scent [SEP] fragrance free [SEP] lightly scented [SEP] Daily Moisturizing Body Lotion for Dry and Sensitive Skin [SEP] "
            "Price: $14.5 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        ),
        "actions": '"click[back to search]", "click[< prev]", "click[12 ounce]", "click[20 ounce]", "click[fragrance free]", "click[lightly scented]", "click[description]", "click[features]", "click[reviews]", "click[buy now]"',
    },
]


def build_prompts(tokenizer, skill_provider):
    """Returns list of (student_prompt_text, teacher_prompt_text)."""
    pairs = []
    for t in TASKS:
        user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=t["task"],
            current_observation=t["obs"],
            available_actions=t["actions"],
        )
        student_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        skill_text = skill_provider.get_privileged_info_from_prompt(user_content)
        teacher_text = f"[Privileged Skill Information]\n{skill_text}\n\n" + student_text
        pairs.append((student_text, teacher_text))
    return pairs


@torch.no_grad()
def response_logits(model, prompt_ids, response_ids, device):
    """Full-vocab logits at each response position. Returns (len_resp, vocab)."""
    input_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0]  # (L, V)
    n_resp = response_ids.size(0)
    # logits predicting response token t live at position (prompt_len + t - 1)
    return logits[-n_resp - 1 : -1].float() / TEMPERATURE


def candidate_entropy(logps_full, cand_ids):
    """Renormalized entropy over candidate set. logps_full: (L, V), cand_ids: (L, K)."""
    cand_logps = torch.gather(logps_full, -1, cand_ids)  # (L, K)
    cand_logps = cand_logps - torch.logsumexp(cand_logps, dim=-1, keepdim=True)
    p = cand_logps.exp()
    return -(p * cand_logps).sum(-1)  # (L,)


def main():
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    )
    model.eval()
    skill_provider = SkillProvider(skills_dir=SKILLS_DIR, skill_all=False)

    pairs = build_prompts(tokenizer, skill_provider)

    all_records = []
    boosted_examples = []

    for pi, (student_text, teacher_text) in enumerate(pairs):
        student_prompt_ids = torch.tensor(tokenizer.encode(student_text, add_special_tokens=False))
        teacher_prompt_ids = torch.tensor(tokenizer.encode(teacher_text, add_special_tokens=False))

        gen = model.generate(
            input_ids=student_prompt_ids.unsqueeze(0).to(device),
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=1.0,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=N_SAMPLES_PER_PROMPT,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for si in range(gen.size(0)):
            response_ids = gen[si, student_prompt_ids.size(0):].cpu()
            # strip trailing pad
            eos_pos = (response_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                response_ids = response_ids[: eos_pos[0] + 1]
            if response_ids.numel() < 4:
                continue

            s_logits = response_logits(model, student_prompt_ids, response_ids, device)
            t_logits = response_logits(model, teacher_prompt_ids, response_ids, device)
            s_logps = F.log_softmax(s_logits, dim=-1)
            t_logps = F.log_softmax(t_logits, dim=-1)

            cand_ids = t_logps.topk(TOP_K, dim=-1).indices  # (L, K) teacher candidates
            h_t = candidate_entropy(t_logps, cand_ids)
            h_s = candidate_entropy(s_logps, cand_ids)
            delta_h = (h_t - h_s)

            realized = response_ids.to(device)
            t_top1 = t_logps.argmax(-1) == realized
            s_top1 = s_logps.argmax(-1) == realized
            eligible = t_top1 & s_top1

            rec = {
                "prompt_idx": pi,
                "n_tokens": int(realized.numel()),
                "t_top1_rate": t_top1.float().mean().item(),
                "s_top1_rate": s_top1.float().mean().item(),
                "both_top1_rate": eligible.float().mean().item(),
                "delta_h_all_mean": delta_h.mean().item(),
                "delta_h_eligible_mean": delta_h[eligible].mean().item() if eligible.any() else None,
                "delta_h_eligible_std": delta_h[eligible].std().item() if eligible.sum() > 1 else None,
                "frac_eligible_pos_dh": (delta_h[eligible] > 0).float().mean().item() if eligible.any() else None,
                "boost_frac_of_all": (eligible & (delta_h > 0)).float().mean().item(),
                "delta_h_eligible_q": (
                    torch.quantile(delta_h[eligible], torch.tensor([0.1, 0.5, 0.9, 0.99]).to(device)).tolist()
                    if eligible.sum() > 4 else None
                ),
            }
            all_records.append(rec)

            # collect a few example boosted tokens with context
            boost_idx = (eligible & (delta_h > 0.3)).nonzero(as_tuple=True)[0]
            for bi in boost_idx[:5].tolist():
                lo = max(0, bi - 6)
                boosted_examples.append({
                    "context": tokenizer.decode(response_ids[lo:bi]),
                    "token": tokenizer.decode(response_ids[bi:bi + 1]),
                    "delta_h": round(delta_h[bi].item(), 3),
                    "h_teacher": round(h_t[bi].item(), 3),
                    "h_student": round(h_s[bi].item(), 3),
                })
        print(f"prompt {pi}: done ({len(all_records)} responses so far)", flush=True)

    def agg(key):
        vals = [r[key] for r in all_records if r[key] is not None]
        return sum(vals) / max(len(vals), 1)

    summary = {
        "n_responses": len(all_records),
        "total_tokens": sum(r["n_tokens"] for r in all_records),
        "teacher_top1_rate": agg("t_top1_rate"),
        "student_top1_rate": agg("s_top1_rate"),
        "both_top1_rate": agg("both_top1_rate"),
        "delta_h_all_mean": agg("delta_h_all_mean"),
        "delta_h_eligible_mean": agg("delta_h_eligible_mean"),
        "frac_eligible_pos_dh": agg("frac_eligible_pos_dh"),
        "boost_frac_of_all_tokens": agg("boost_frac_of_all"),
    }
    print(json.dumps(summary, indent=2))
    with open(OUT_PATH, "w") as f:
        json.dump({"summary": summary, "records": all_records, "boosted_examples": boosted_examples[:60]}, f, indent=2, ensure_ascii=False)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
