from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

result = generator(
    "Hello, I'm a language model,",
    max_new_tokens=30,                 # лучше, чем max_length
    num_return_sequences=5,
    truncation=True                    # явно включаем усечение
)

for i, r in enumerate(result, 1):
    print(f"{i}. {r['generated_text']}\n")