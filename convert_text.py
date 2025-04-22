from datasets import load_from_disk
import re
from transformers import AutoTokenizer
from datasets import Dataset
from spy import Transformer

config = {
    "tokenizer_dir": "Salesforce/codegen-350M-multi",
    "cache_dir": "./cached",
    "local_files_only": True,
    "code": '''def mult(numbers):
    if len(numbers) == 0:
        raise ValueError
            
        output = 1
        for num in numbers:
            output *= num

        return output

    value = mult([5, 10, 4])
    print(value)
    '''
}

dataset = load_from_disk("./cached/flytech_original")
print("Original: \n" + config["code"])

def convert_to_spy(sample):
    output = sample

    try:
        spy_code = transformer.parse(output)
        return spy_code
    except (ValueError, RecursionError):
        return None
    

transformer = Transformer()
    
code = convert_to_spy(config["code"])
print("Converted: \n" + code)