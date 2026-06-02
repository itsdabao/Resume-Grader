import os
import sys


def main() -> None:
    sys.path.insert(0, os.getcwd())

    from app.services.agentic.preprocess import is_toxic, preprocess_query

    query = "Trung tâm có các khóa học nào?"
    print(f"Query: {query}")
    toxic = is_toxic(query)
    print(f"Is Toxic: {toxic}")

    result = preprocess_query(query)
    print(f"Preprocess Result: {result}")


if __name__ == "__main__":
    main()
