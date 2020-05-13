import spacy
import torch
import numpy
from spacy.vectors import Vectors
from utils import load_cached_embeddings

# embeddings_dict = load_cached_embeddings(".\\glove\\glove.6B.300d.txt")
model = "en_core_web_sm"

nlp = spacy.load("en_core_web_sm")
# nlp.vocab.prune_vectors(105000)
# nlp.vocab.vectors.from_glove("./glove/glove.6B.300d.txt")
# nlp.vocab.vectors.from_glove(".\\glove")

# vector_table = numpy.zeros((3, 300), dtype="f")
# vectors = Vectors(data=vector_table, keys=["dog", "cat", "orange"])
# vectors.data = torch.Tensor(vectors.data).cuda(0)

# print(vectors.data)

# https://bpben.github.io/2019/09/18/spacy_pytorch_walkthrough/
print(nlp.pipe_names)
print(f"Loaded model '{model}'")
# textcat = nlp.create_pipe(
#     "pytt_textcat",
#     config={"architecture": "softmax_last_hidden", "words_per_batch": 50},
# )

# sentencizer = nlp.create_pipe("sentencizer")
# nlp.add_pipe(sentencizer)

texts = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]
disabled = nlp.disable_pipes(["tagger", "parser"])
# for doc in nlp.pipe(texts, disable=["tagger", "parser"]):
for doc in nlp.pipe(texts):
    # Do something with the doc here
    print([(ent.text, ent.label_) for ent in doc.ents])
disabled.restore()
