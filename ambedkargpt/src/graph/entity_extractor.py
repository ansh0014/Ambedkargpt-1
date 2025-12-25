

import json
from typing import List, Dict
from pathlib import Path

import spacy


class EntityExtractor:
    """
    Extracts entities from semantic chunks using spaCy NER.
    Entities will later become nodes in the Knowledge Graph.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        allowed_labels: List[str] = None,
    ):
        self.nlp = spacy.load(model_name)

       
        self.allowed_labels = allowed_labels or [
            "PERSON",
            "ORG",
            "GPE",
            "DATE",
            "LAW",
            "EVENT",
        ]

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from a single text chunk.
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.allowed_labels:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                    }
                )

        return entities

    def process_chunks(self, chunks_path: str, output_path: str):
        """
        Load chunks.json, extract entities for each chunk,
        and save enriched chunks.
        """
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            chunk["entities"] = self.extract_entities(chunk["text"])

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        print(f"[OK] Extracted entities for {len(chunks)} chunks")



if __name__ == "__main__":
    extractor = EntityExtractor()
    extractor.process_chunks(
        chunks_path="data/processed/chunks.json",
        output_path="data/processed/chunks.json",
    )
