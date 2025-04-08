from .lora_skeleton import LoRALinear, process_sequences, evaluate
from .qwen import load_qwen
from .preprocessor import load_and_preprocess
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# 1. Prepare model and optimizer
def prepare_model(lora_rank=4, learning_rate=1e-5):
    model, tokenizer = load_qwen()
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)
    
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), 
        lr=learning_rate
    )
    return model, tokenizer, optimizer


def tokenise_data(train_texts, val_texts, tokenizer, max_ctx_length=512):
    train_ids = process_sequences(train_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2)
    val_ids = process_sequences(val_texts, tokenizer, max_ctx_length, stride=max_ctx_length)
    return train_ids, val_ids

# 2. Turn token ids into a DataLoader
def prepare_data_loader(train_ids, batch_size=4):
    train_loader = DataLoader(TensorDataset(train_ids), batch_size=batch_size, shuffle=True)
    return train_loader



# 3. Training loop
def train(model, optimizer, train_loader, val_ids, 
          train_steps=6000, val_every=200, batch_size=4):
    
    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    train_losses = []
    val_losses = []

    model.train()
    steps = 0
    cumulative_loss = 0.0
    count = 0

    while steps < train_steps:
        progress = tqdm(train_loader, desc=f"Training @ step {steps}")
        for (batch,) in progress:
            optimizer.zero_grad()
            loss = model(batch, labels=batch).loss
            accelerator.backward(loss)
            optimizer.step()
            steps += 1

            cumulative_loss += loss.item()
            count += 1

            progress.set_postfix(loss=loss.item())

            if steps % val_every == 0:
                train_losses.append(cumulative_loss / count)
                model.eval()
                val_loss = evaluate(model, val_ids, batch_size)
                val_losses.append(val_loss)
                model.train()
                cumulative_loss = 0.0
                count = 0

            if steps >= train_steps:
                break

    if steps % val_every != 0:
        train_losses.append(cumulative_loss / count)
        model.eval()
        val_loss = evaluate(model, val_ids, batch_size)
        val_losses.append(val_loss)
        model.train()

    return train_losses, val_losses, loss.item()


# 4. Generate Predictions
  
# ====== Generate Predictions (with tokenization inside) ======
@torch.no_grad()
def generate_predictions(prompt_string, tokenizer, model, device,
                         max_new_tokens=1200, num_samples=1, do_sample=False):
    tokens = tokenizer(prompt_string, return_tensors="pt")["input_ids"].to(device)


    if not do_sample and num_samples != 1:
        raise ValueError("num_return_sequences > 1 requires do_sample=True")

    outputs = model.generate(
        input_ids=tokens,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.eos_token_id,
        temperature=None,
        top_p = None,
        top_k = None
    )
    

    return [output[len(tokens[0]):].tolist() for output in outputs]

