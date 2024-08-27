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
    model_name="ozzyable/log-summerizer-tinyllama-v3",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True
)

FastLanguageModel.for_inference(model)

def summarize(transactions):
    formatted_inputs = [
        alpaca_prompt.format(
            "you are a transaction interpreter , you receive transactions that are writen for the banking context & you extract valuable data them in json format: {'transaction_channel': (Transfer, Online Payement, Card Payement, Bank fee, Deposit), 'other_party_name': name of the sender or receiver, 'info': motif or reason of the transaction, if there is no motif just leave it blank}", # instruction
            f"{transaction}", # input
            ""
        )
        for transaction in transactions
    ]
    
    inputs = tokenizer(
        formatted_inputs,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    result = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)

    for i in range(len(transactions)):
        transactions[i]['summary'] = result[i][result.find("<bor>") + 5:].strip()

    return transactions

async def process_input(input):
    transactions = input.get('transaction', [])
    return summarize(transactions)

async def handler(event):
    return await process_input(event['input'])

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})