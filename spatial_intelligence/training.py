"""Inspectable synchronous data-parallel reference trainer with real optimizer steps."""
import copy
import json
import random
from pathlib import Path

import torch

from .backends import create
from .data import Mixture, content_fingerprint, key, leakage, load_samples, model_input
from .evaluation import score_one
from .io import environment, file_digest, symbol, write_json
from .objectives import group_advantages, kl_loss, policy_loss, sft_loss, token_logps


def reward(response, row, cfg):
    if cfg["reward_plugin"]:
        value = symbol(cfg["reward_plugin"])(response, row, cfg)
        return float(value)
    return float(score_one(response, row, cfg["numeric_atol"], cfg["numeric_rtol"])["correct"])


def train(cfg):
    from .distributed import Distributed
    context=Distributed(cfg)
    try:return _train(context.cfg,context)
    finally:context.close()


def _train(cfg,context):
    t = cfg["train"]
    batch_size=t.get("batch_size",1)
    if not isinstance(batch_size,int) or batch_size<1:raise ValueError("train.batch_size must be a positive integer")
    local_examples=batch_size*t["gradient_accumulation"]
    mode = t["mode"]
    if mode not in {"sft", "rl", "opd", "opsd", "offline_kd"}:
        raise ValueError("Unknown training mode")
    if t["steps"] < 1 or t["gradient_accumulation"] < 1 or t["learning_rate"] <= 0:
        raise ValueError("Positive steps, gradient accumulation and learning rate required")
    if mode == "rl" and t["group_size"] < 2:
        raise ValueError("Group-relative RL requires group_size >= 2")
    generation = cfg["protocol"]["generation"]
    if mode in {"rl", "opd", "opsd"}:
        if not generation.get("do_sample") or generation.get("top_p", 1.0) != 1.0 or generation.get("top_k", 0) != 0:
            raise ValueError("Reference on-policy training requires do_sample=true, top_p=1, top_k=0")
        if generation.get("repetition_penalty", 1.0) != 1 or generation.get("num_beams", 1) != 1:
            raise ValueError("On-policy probabilities require unmodified categorical sampling")
    out = Path(cfg["output"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; select a new run directory")
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    mixture = Mixture(cfg["data"]["train"], t["seed"])
    all_train = [r for ds in mixture.datasets for r in ds]
    heldout = [r for p in cfg["data"]["heldout"] for r in load_samples(p)]
    if not heldout:
        raise ValueError("Declare heldout manifests to audit leakage before training")
    if any(r["split"] == "train" for r in heldout):
        raise ValueError("Heldout manifest contains split=train")
    overlap = leakage(all_train, heldout)
    if any(overlap.values()):
        raise ValueError(f"Train/heldout overlap: { {k:len(v) for k,v in overlap.items()} }")
    context.barrier()
    out.mkdir(parents=True, exist_ok=True)
    def record_json(path,value):
        if context.main:write_json(path,value)
    record_json(out / "config.json", cfg)
    record_json(out / "environment.json", environment())
    record_json(out / "data_audit.json", {"overlap": overlap,
        "train_hash": content_fingerprint(all_train), "heldout_hash": content_fingerprint(heldout),
        # Includes offline-KD teacher responses and provenance stored in metadata.
        "train_manifest_sha256": {entry["path"]: file_digest(entry["path"]) for entry in cfg["data"]["train"]},
        "heldout_manifest_sha256": {path: file_digest(path) for path in cfg["data"]["heldout"]},
        "train_without_scene_id": sum(r["scene_id"] is None for r in all_train),
        "warning": "Exact overlap check only; perceptually similar scenes and teacher pretraining contamination are not excluded"})
    student = create(cfg["model"], training=True)
    context.broadcast_parameters(student.model)
    teacher = None
    if mode in {"rl", "opsd"}:
        teacher = copy.deepcopy(student)
        teacher.model.requires_grad_(False)
        teacher.model.eval()
    elif mode == "opd":
        teacher = create(cfg["teacher"], training=False)
    if teacher and student.tokenizer_fingerprint() != teacher.tokenizer_fingerprint():
        raise ValueError("Token-level KL requires identical tokenizer semantics. Use offline_kd for heterogeneous teachers.")
    params = [p for p in student.model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=t["learning_rate"], weight_decay=t["weight_decay"])
    temperature = generation.get("temperature", 1.0)
    counts = {}
    record_json(out / "model_info.json", {"trainable_parameters": sum(p.numel() for p in params),
        "total_parameters": sum(p.numel() for p in student.model.parameters()),
        "tokenizer_hash": student.tokenizer_fingerprint(), "teacher_mode": mode,
        "student_revision": getattr(student, "resolved_revision", None),
        "teacher_revision": getattr(teacher, "resolved_revision", None),
        "student_identity":student.identity() if hasattr(student,"identity") else None,
        "teacher_identity":teacher.identity() if teacher and hasattr(teacher,"identity") else None,
        "world_size":context.world,"per_rank_batch_size":batch_size,"gradient_accumulation":t["gradient_accumulation"],
        "global_prompt_batch":local_examples*context.world,"microbatch_execution":"sequential_examples"})
    with (out / ("metrics.jsonl" if context.main else f"metrics.rank{context.rank}.jsonl")).open("w", encoding="utf-8") as log:
        for step in range(t["steps"]):
            optimizer.zero_grad(set_to_none=True)
            step_loss, rewards_log, tokens, sampled = 0.0, [], 0, []
            for micro in range(local_examples):
                # All ranks draw the same global stream, then select disjoint positions.
                global_rows=[mixture.sample() for _ in range(context.world)]
                row = global_rows[context.rank]
                sampled.append(key(row))
                counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
                inp = model_input(row)
                batch = student.encode(inp, cfg["protocol"])
                if mode in {"sft", "offline_kd"}:
                    target = row["metadata"].get("teacher_response") if mode == "offline_kd" else row["answer"]
                    if not target:
                        raise ValueError("offline_kd requires metadata.teacher_response from an actual teacher rollout")
                    if mode == "sft":
                        target = "<answer>" + target + "</answer>"
                    ids = student.response_ids(target)
                    loss = sft_loss(student.response_logits(batch, ids), ids)
                    (loss/local_examples).backward()
                    step_loss += float(loss.detach()) / local_examples
                    tokens += ids.numel()
                else:
                    group_size = t["group_size"] if mode == "rl" else 1
                    rollouts = []
                    # Roll out the entire group before computing any policy gradient.
                    for g in range(group_size):
                        seed = t["seed"] + ((step*local_examples+micro)*context.world+context.rank)*group_size + g
                        ids = student.sample_ids(batch, generation, seed)
                        if ids.numel() == 0:
                            raise ValueError("Empty rollout")
                        response = student.tokenizer.decode(ids[0], skip_special_tokens=True)
                        r = reward(response, row, t) if mode == "rl" else 0.0
                        if not torch.isfinite(torch.tensor(r)):
                            raise ValueError("Nonfinite reward")
                        with torch.no_grad():
                            old = token_logps(student.response_logits(batch, ids), ids, temperature) if mode == "rl" else None
                        rollouts.append((ids, r, old))
                    advantages = group_advantages([r for _, r, _ in rollouts])
                    for g, (ids, r, old) in enumerate(rollouts):
                        privileged = row["answer"] if mode == "opsd" and t["privileged_answer"] else None
                        teacher_batch = teacher.encode(inp, cfg["protocol"], privileged=privileged)
                        with torch.no_grad():
                            teacher_logits = teacher.response_logits(teacher_batch, ids.to(teacher.device))
                        logits = student.response_logits(batch, ids)
                        if mode == "rl":
                            loss = policy_loss(token_logps(logits, ids, temperature), old,
                                token_logps(teacher_logits, ids.to(teacher.device), temperature).to(student.device),
                                advantages[g].to(student.device), algorithm=t["algorithm"], clip=t["clip"], beta=t["beta"])
                            rewards_log.append(r)
                        else:
                            loss = kl_loss(logits, teacher_logits, direction=t["kl_direction"], temperature=t["distill_temperature"])
                            if t["sft_weight"] > 0:
                                gold = student.response_ids("<answer>"+row["answer"]+"</answer>")
                                loss = loss + t["sft_weight"] * sft_loss(student.response_logits(batch, gold), gold)
                        scale = group_size * local_examples
                        (loss/scale).backward()
                        step_loss += float(loss.detach())/scale
                        tokens += ids.numel()
                        del logits, teacher_logits, loss
            context.average_gradients(params)
            norm = torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"])
            if not context.all_finite(bool(torch.isfinite(norm)) and bool(torch.isfinite(torch.tensor(step_loss)))):
                raise FloatingPointError("Nonfinite gradient/loss; run aborted without optimizer step")
            optimizer.step()
            record = {"step": step+1, "loss": step_loss, "grad_norm": float(norm),
                      "completion_tokens": tokens, "sample_ids": sampled,
                      "mean_reward": sum(rewards_log)/len(rewards_log) if rewards_log else None,
                      "mixture_draws": dict(counts)}
            global_records=context.objects(record)
            if context.main:
                record={**record,"loss":sum(r["loss"] for r in global_records)/context.world,
                        "completion_tokens":sum(r["completion_tokens"] for r in global_records),
                        "sample_ids":[sid for r in global_records for sid in r["sample_ids"]],
                        "mixture_draws":{ds:sum(r["mixture_draws"].get(ds,0) for r in global_records) for ds in {k for r in global_records for k in r["mixture_draws"]}},
                        "mean_reward":sum(r["mean_reward"] for r in global_records)/context.world if rewards_log else None,
                        "global_prompt_batch":local_examples*context.world}
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            log.flush()
            if context.main:print(json.dumps(record, ensure_ascii=False), flush=True)
            if mode == "opsd" and t["teacher_refresh_steps"] > 0 and (step+1) % t["teacher_refresh_steps"] == 0:
                teacher.model.load_state_dict(student.model.state_dict())
                teacher.model.requires_grad_(False)
            if context.main and t["save_every"] > 0 and (step+1) % t["save_every"] == 0:
                student.save(out / f"checkpoint-{step+1}")
        context.barrier()
        if context.main:student.save(out / "final")
        context.barrier()
    record_json(out / "completion.json", {"status": "optimizer_training_completed", "steps": t["steps"],
        "mode": mode, "world_size":context.world, "global_prompt_batch":local_examples*context.world, "checkpoint": str(out / "final"), "exact_resume_supported": False})
    return str(out / "final")
