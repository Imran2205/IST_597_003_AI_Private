import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk
from nltk.tokenize import word_tokenize


def load_pretrained_model():
    """Load pre-trained Word2Vec model from HuggingFace"""
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', model_max_length=512)
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    return model, tokenizer


def get_sentence_embedding(text, model, tokenizer):
    """Generate embedding for text using the pre-trained model"""
    # Tokenize and encode
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


def compare_strings(str1, str2):
    """Compare two strings using pre-trained embeddings"""
    # Load model and tokenizer
    model, tokenizer = load_pretrained_model()
    # Get embeddings
    emb1 = get_sentence_embedding(str1, model, tokenizer)
    emb2 = get_sentence_embedding(str2, model, tokenizer)
    # Calculate similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity


def main():
    # Example usage
    string1 = "sustain"  # , bunny, sinful, cheerful
    # string2 = "A fast brown fox leaps above a sleepy canine"
    strings = ["preserve", "angry", "rabbits", "gleeful", "corrupt", "TVs"]
    for string2 in strings:
        similarity_score = compare_strings(string1, string2)
        print(f"Similarity score between {string1} and {string2}: {similarity_score:.4f}")


if __name__ == "__main__":
    main()
