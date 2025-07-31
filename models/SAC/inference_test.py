from promodel import PROMODEL

def main():
    model = PROMODEL()
    prompt = input("🔍 Enter your prompt: ").strip()
    response = model.generate_response(prompt)
    print(f"💬 LLM Response: {response}")
    liked = input("👍 Was the response helpful? (y/n): ").strip().lower()
    metrics = model.feedback(prompt, liked == 'y')

    print("\n📊 Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

if __name__ == "__main__":
    main()