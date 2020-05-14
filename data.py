"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import pickle
from random import shuffle

import spacy
import torch
from torch.utils.data import Dataset

from utils import cuda, load_dataset

spacy.require_gpu()

DEBUG = False

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """

    def __init__(self, samples, vocab_size):
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for (_, passage, question, _, _) in samples:
            for token in itertools.chain(passage, question):
                vocab[token.lower()] += 1
        top_words = [
            word
            for (word, _) in sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words

    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id) for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN) for token_id in token_ids
        ]


class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). Passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """

    def __init__(self, args, path, name):
        self.processor = spacy.load("en_core_web_sm")
        self.args = args
        self.name = name
        self.meta, self.elems = load_dataset(path)
        self.samples = self._create_samples()
        self.tokenizer = None
        self.batch_size = args.batch_size if "batch_size" in args else 1
        self.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        )

    def _create_samples(self):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """

        try:
            with open(self.name + ".pkl", "rb") as f:
                print(f"using cached {self.name} from pickle bin")
                return pickle.load(f)
        except EnvironmentError:
            pass

        samples = []
        self.processor.add_pipe(self.processor.create_pipe("sentencizer"))
        disabled = self.processor.disable_pipes(["tagger", "parser"])

        # batch process entries with spacy
        contexts = list(self.processor.pipe([elem["context"] for elem in self.elems]))
        # qas_contexts = list(
        #     self.processor.pipe(
        #         qa["question"] for elem in self.elems for qa in elem["qas"]
        #     )
        # )

        for (el_i, elem) in enumerate(self.elems):
            if not el_i % 100:
                print(f"processing elem {el_i} of {len(self.elems)}")

            # NOTE: spacy - process passage
            # passage_context = self.processor(elem["context"])
            passage_context = contexts[el_i]
            qas_context = list(
                self.processor.pipe(qa["question"] for qa in elem["qas"])
            )

            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            for (qa_id, qa) in enumerate(elem["qas"]):
                qid = qa["qid"]
                question = [token.lower() for (token, offset) in qa["question_tokens"]][
                    : self.args.max_question_length
                ]

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.
                answers = qa["detected_answers"]
                answer_start, answer_end = answers[0]["token_spans"][0]

                # NOTE: spacy - begin focussing passage to target area
                # question_context = self.processor(qa["question"])
                question_context = qas_context[qa_id]

                # view passage
                DEBUG and print()
                DEBUG and print()
                DEBUG and print(elem["context"])

                # view passage named entities
                for ent in passage_context.ents:
                    DEBUG and print(
                        ent.text,
                        ent.start_char,
                        ent.end_char,
                        ent.label_,
                        spacy.explain(ent.label_),
                    )

                # view question named entities
                DEBUG and print()
                DEBUG and print(qa["question"])
                for ent in question_context.ents:
                    DEBUG and print(
                        ent.text,
                        ent.start_char,
                        ent.end_char,
                        ent.label_,
                        spacy.explain(ent.label_),
                    )

                token_id = 0
                answer_start_new = answer_start
                answer_end_new = answer_end
                matched_sents = []

                # update start & end marks to reflect trimmed passage
                for sent in passage_context.sents:
                    # get label of each entity in each q & passage
                    q_ents = [q_ent.label for q_ent in question_context.ents]
                    s_ents = [s_ent.label for s_ent in sent.ents]

                    # only keep sentences w overlapping entities w question
                    if bool(set(q_ents).intersection(s_ents)):
                        matched_sents.append(sent)
                    elif token_id < answer_start:
                        answer_start_new -= len(sent)
                        answer_end_new -= len(sent)
                    token_id += len(sent)

                # convert to list of tokens like original passage
                unique_passage = [
                    token.text.lower() for sent in matched_sents for token in sent
                ][: self.args.max_context_length]

                # view original passage & trimmed
                DEBUG and print(
                    [token.lower() for (token, offset) in elem["context_tokens"]][
                        : self.args.max_context_length
                    ]
                )
                DEBUG and print(unique_passage)

                # don't use examples where this process fails
                if (
                    answer_start_new > -1
                    and answer_end_new > -1
                    and len(unique_passage) > 0
                ):
                    samples.append(
                        (
                            qid,
                            unique_passage,
                            question,
                            answer_start_new,
                            answer_end_new,
                        )
                    )
                else:
                    samples.append(
                        (
                            qid,
                            [
                                token.lower()
                                for (token, offset) in elem["context_tokens"]
                            ][: self.args.max_context_length],
                            question,
                            answer_start,
                            answer_end,
                        )
                    )

        disabled.restore()

        # save processed samples
        with open(self.name + ".pkl", "wb") as f:
            pickle.dump(samples, f)

        return samples

    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError("error: no tokenizer registered")

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        passages = []
        questions = []
        start_positions = []
        end_positions = []
        for idx in example_idxs:
            # Unpack QA sample and tokenize passage/question.
            qid, passage, question, answer_start, answer_end = self.samples[idx]

            # Convert words to tensor.
            passage_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(passage))
            question_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(question))
            answer_start_ids = torch.tensor(answer_start)
            answer_end_ids = torch.tensor(answer_end)

            # Store each part in an independent list.
            passages.append(passage_ids)
            questions.append(question_ids)
            start_positions.append(answer_start_ids)
            end_positions.append(answer_end_ids)

        return zip(passages, questions, start_positions, end_positions)

    def _create_batches(self, generator, batch_size):
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages = []
            questions = []
            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            max_passage_length = 0
            max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages.append(current_batch[ii][0])
                questions.append(current_batch[ii][1])
                start_positions[ii] = current_batch[ii][2]
                end_positions[ii] = current_batch[ii][3]
                max_passage_length = max(max_passage_length, len(current_batch[ii][0]))
                max_question_length = max(
                    max_question_length, len(current_batch[ii][1])
                )

            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            padded_passages = torch.zeros(bsz, max_passage_length)
            padded_questions = torch.zeros(bsz, max_question_length)
            # Pad passages and questions
            for iii, passage_question in enumerate(zip(passages, questions)):
                passage, question = passage_question
                padded_passages[iii][: len(passage)] = passage
                padded_questions[iii][: len(question)] = question

            # Create an input dictionary
            batch_dict = {
                "passages": cuda(self.args, padded_passages).long(),
                "questions": cuda(self.args, padded_questions).long(),
                "start_positions": cuda(self.args, start_positions).long(),
                "end_positions": cuda(self.args, end_positions).long(),
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size,
        )

    def register_tokenizer(self, tokenizer):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)
