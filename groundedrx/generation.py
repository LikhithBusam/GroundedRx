"""Answer generation via the locally-loaded model (config.MODEL_NAME).

Ported from GroundedRx_Colab.ipynb Component 5, Cell 2. GPU-only -- imports
torch and calls the model, so this module is not part of the CPU-testable
tier (see tests/ for what is).
"""

from . import config


def generate_answer(query: str, context: str, language: str) -> dict:
    """
    Generate a medical answer using the loaded model.

    Context-only prompting (prevents hallucination) with separate AR/EN
    prompt templates. Must go through `apply_chat_template`, not a raw
    `tokenizer(prompt)` call -- true for any instruct-tuned chat model, and
    skipping it previously caused the model to hallucinate literal
    role-turn text (see CLAUDE.md).
    """
    import torch

    from .resources import get_model, get_tokenizer

    tokenizer = get_tokenizer()
    model = get_model()

    template = config.PROMPT_AR if language == "ar" else config.PROMPT_EN
    context_truncated = context[:2000]  # safe VRAM limit

    prompt = template.format(context=context_truncated, query=query)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    # Greedy decoding -- deterministic, matches the evaluation judge calls.
    # Leading hypothesis (not fully confirmed) is that sampling under 4-bit
    # NF4 quantization draws noisy low-probability tokens more often for
    # Arabic than English; greedy removes that risk outright. See CLAUDE.md
    # "Arabic generation quality investigation".
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=450,
            do_sample=False,
            repetition_penalty=1.1,  # unrelated to sampling noise -- guards against greedy repetition loops
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "language": language,
        "input_tokens": input_len,
        "output_tokens": len(new_tokens),
        # the TRUNCATED context -- the gate must judge grounding against what
        # the model actually saw, not the full context it never received.
        "context_used": context_truncated,
    }
