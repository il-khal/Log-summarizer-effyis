import torch._dynamo
torch._dynamo.config.suppress_errors = True

import runpod
from unsloth import FastLanguageModel
import asyncio

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:<bor>
"""

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ozzyable/log-summerizer-gemma-2b-it-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True
)

FastLanguageModel.for_inference(model)

async def summarize(transaction):
    description = transaction.get("descriptionTransaction", "")
    
    inputs = tokenizer(
        alpaca_prompt.format(
            "you are a transaction interpreter, you receive transactions that are written for the banking context & you extract valuable data from them in JSON format: {'transaction_channel': (Transfer, Online Payment, Card Payment, Bank fee, Deposit), 'other_party_name': name of the sender or receiver, 'info': motif or reason of the transaction, if there is no motif just leave it blank}",
            description,
            ""
        ),
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    transaction['summary'] = result[result.find("<bor>") + 5:].strip()
    return transaction

async def process_input(input):
    transactions = input.get('transaction', [])
    return await asyncio.gather(*map(summarize, transactions))

async def handler(event):
    return await process_input(event['input'])

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})