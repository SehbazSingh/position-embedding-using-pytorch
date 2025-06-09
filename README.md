# Position Embedding with GPT-2 Tokenizer (PyTorch)

This project demonstrates how to build a GPT-style token and positional embedding pipeline using the `tiktoken` GPT-2 tokenizer and PyTorch. It includes a custom dataset class for text chunking, a dataloader, and embedding layers for tokens and positions.

## 📂 File

- `position-embedding.py`: The main Python script containing dataset creation, tokenization, and embedding logic.

## 🚀 Features

- GPT-2 tokenizer integration with `tiktoken`
- Custom `Dataset` class with sliding window support
- PyTorch `Embedding` layers for tokens and positions
- Combines token and positional embeddings
- Works on any plain text file (e.g., `the-verdict.txt`)

## 🛠️ Requirements

- Python 3.7+
- PyTorch
- tiktoken

Install dependencies:

```bash
pip install torch tiktoken
```
📌 Usage
Place your text file in the same directory (e.g., the-verdict.txt)

Run the script:
```
python position-embedding.py
```
Sample output:
```
torch.Size([8, 4, 256])
```
This output shows the shape of the combined token and positional embeddings:

8 sequences per batch

4 tokens per sequence

256-dimensional embeddings

📚 How it Works
Tokenizer: Uses GPT-2 tokenizer from OpenAI's tiktoken.

Dataset: Splits long token streams into fixed-length chunks with optional overlap (stride).

Dataloader: Batches the chunks for training.

Embeddings:

Token embeddings are looked up from vocabulary.

Positional embeddings represent each token's position in the sequence.

Final input embeddings are the sum of the above.

🧠 Example Code Snippet
```
pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
input_embeddings = token_embeddings + pos_embeddings
```
📄 License
This project is licensed under the MIT License.
