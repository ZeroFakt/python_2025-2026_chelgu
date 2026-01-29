from search import ICD10Searcher


def main():
    searcher = ICD10Searcher("data/icd10_ru.xlsx")

    while True:
        text = input("Введите диагноз (RU): ").strip()
        if not text:
            break

        results = searcher.search(text, top_k=3)
        for r in results:
            print(f"{r['code']} | {r['description']} | score={r['score']:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
