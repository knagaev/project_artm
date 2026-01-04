# Токенизаторы как переиспользуемые компоненты
def simple_tokenizer(text: str) -> list[str]:
    return text.lower().split()
