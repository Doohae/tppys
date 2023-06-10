from transformers import AutoTokenizer


class Tokenizer:
    def __init__(
        self,
        tokenizer_name: str,
        revision: str,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, revision=revision
        )

    def encode(self, text: str):
        encoded = self.tokenizer.encode(text)
        return encoded
