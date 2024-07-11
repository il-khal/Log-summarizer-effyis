import torch._dynamo
torch._dynamo.config.suppress_errors = True # suppressing errors : safe  mesure for unexpected torch errors


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
    # othermodels are at https://huggingface.co/ozzyable
    model_name="ozzyable/log-summerizer-gemma-2b-it-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True
    )

FastLanguageModel.for_inference(model)

def summarize(transaction: str):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "you are a transaction interpreter , you receive transactions that are writen for the banking context & you extract valuable data them in json format: {'transaction_channel': (Transfer, Online Payement, Card Payement, Bank fee, Deposit), 'other_party_name': name of the sender or receiver, 'info': motif or reason of the transaction, if there is no motif just leave it blank}", # instruction
                transaction,
                ""
            )
        ],
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    substring = "{'transaction_channel': '"
    start_index = result.find(substring)

    formatted_output = result[start_index:]

    return formatted_output


def process_input(input):
    transactions = input.get('transaction', [])
    results = []
    for transaction in transactions:
        results.append(summarize(transaction))

    return results

def handler(event):
    # "event" is the input the pod receives
    # returns a list of strings, each string contains the unparsed json for each transaction
    return process_input(event['input'])

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
