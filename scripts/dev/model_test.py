def main() -> None:
    from llama_cpp import Llama

    model_path = "models/qwen2.5-3b-instruct-q4_k_m.gguf"

    print(">>> Đang khởi động AI (llama.cpp)…")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=True,
    )

    question = "Chào bạn, hãy giới thiệu ngắn gọn về bản thân bằng tiếng Việt."
    print(f"\nUser: {question}")

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Bạn là trợ lý AI hữu ích."},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )

    print(f"AI: {output['choices'][0]['message']['content']}")


if __name__ == "__main__":
    main()
