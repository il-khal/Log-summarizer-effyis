import torch._dynamo
torch._dynamo.config.suppress_errors = True


import runpod
from unsloth import FastLanguageModel

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ozzyable/log-summ-gemma-v2",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=False
    )

FastLanguageModel.for_inference(model)

def summarize(transactions: list):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "you are a transaction interpreter , you receive transactions that are writen for the banking context & you extract the transaction category: other, financial, pension, reimbursement and salary", # instruction
                transaction,
                ""
            ) for transaction in transactions
        ],
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    s = []
    
    for transaction in results:
        substring = "{'transaction_channel': '"
        start_index = transaction.find(substring)

        formatted_output = transaction[start_index:]
        s.append(formatted_output)

    return s


def process_input(input):
    transactions = input.get('transaction', [])
    results = summarize(transactions)

    return results

def handler(event):
    return process_input(event['input'])

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
