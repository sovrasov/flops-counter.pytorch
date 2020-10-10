# ! pip install transformers
# ! pip install ptflops

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from ptflops import get_model_complexity_info

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


def my_input_constructor(input_shape):
    inp_seq = ""
    for _ in range(input_shape[1] - 2):  # there are two special tokens [CLS] and [SEP]
        inp_seq += tokenizer.pad_token  # let's use pad token to form a fake sequence for subsequent flops calculation

    inputs = tokenizer([inp_seq] * input_shape[0], padding=True, truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * input_shape[0])
    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


flops_count, params_count = get_model_complexity_info(model, (2, 128), as_strings=True,
                                                      input_constructor=my_input_constructor,
                                                      print_per_layer_stat=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', flops_count))
print('{:<30}  {:<8}'.format('Number of parameters: ', params_count))

## Output:
# Computational complexity:       21.74 GMac
# Number of parameters:           109.48 M
