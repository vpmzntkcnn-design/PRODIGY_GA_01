from transformers import pipeline, set_seed

# Load model
generator = pipeline('text-generation', model='gpt2')

# Fix randomness
set_seed(42)

# Take input
prompt = input("Enter your prompt: ")

# Generate text
output = generator(
    prompt,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=2
)

print("\nGenerated Text:\n")
print(output[0]['generated_text'])