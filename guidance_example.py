# guidance library, just like instructor, pydantic_ai, marvin, force LLM to output user specified python type.
# Only works with vllm

@guidance
def extract_person(lm, text):
    lm += "Extract info:\n"
    lm += "{"
    lm += '"name": "' + guidance.gen(stop='"') + '",'
    lm += '"age": ' + guidance.gen(regex=r"\d+") 
    lm += "}"

#Other example.  It will return yes or no.
lm += guidance.select(["yes", "no"])
