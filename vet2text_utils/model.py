import os
import math
import torch
import vec2text

os.environ["OPENAI_API_KEY"] = ""


corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")

from openai import OpenAI
client = OpenAI()
def get_embeddings_openai(text_list, model="text-embedding-ada-002") -> torch.Tensor:
    batches = math.ceil(len(text_list) / 128)
    outputs = []
    for batch in range(batches):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = client.embeddings.create(
            input=text_list_batch,
            model=model,
            encoding_format="float",  # override default base64 encoding...
        )
        outputs.extend([e["embedding"] for e in response["data"]])
    return torch.tensor(outputs)
