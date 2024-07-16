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
    model_name="ozzyable/log-analysis-tinyllama",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=False
    )

FastLanguageModel.for_inference(model)

def summarize(transactions: list):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "you are a log analyzer , you receive logs as inputs and you should analyze these logs and repond with that information the log was infering.", # instruction
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
    
    # extracting the response from the results
    for transaction in results:
        substring = "Response:"
        start_index = transaction.find(substring) + len(substring)

        formatted_output = transaction[start_index:]
        s.append(formatted_output)

    return s


def process_input(input):
    transactions = input.get('logs', [])
    results = summarize(transactions)

    return results

def handler(event):
    # "event" is the input the pod receives
    # returns a list of strings, each string contains the unparsed json for each transaction
    return process_input(event['input'])

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
