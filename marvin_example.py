# Another tool aside from pydantic_ai, instructor, that can make LLM return specified python type.
# Currently only support openAI.

from pydantic import BaseModel
import marvin

class Person(BaseModel):
    name: str
    age: int

@marvin.fn
def extract_person(text: str) -> Person:
    """Extract the person's name and age."""

extract_person("Alice is 31 years old")
# → Person(name="Alice", age=31)

# other examples.
@marvin.fn
def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify sentiment."""

@marvin.fn
def is_spam(text: str) -> bool:
    """Determine whether this message is spam."""

@marvin.fn
def extract_entities(text: str) -> list[str]:
    """Extract named entities."""
