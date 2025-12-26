# DSPy example.  
# With training, DSPy helps improve your prompt.

# Your original prompt.
'''
Given the context and question, provide an answer.

Context: [retrieved passages]
Question: What is photosynthesis?
Answer:
'''

# After training with DSPy, the prompt can be like below.
'''
Given the context and question, provide an answer.

Context: [some passage about cellular respiration]
Question: What is cellular respiration?
Answer: The process by which cells convert glucose into energy

Context: [some passage about nitrogen cycle]  
Question: What is the nitrogen cycle?
Answer: The process by which nitrogen is converted between different chemical forms

Context: [retrieved passages about photosynthesis]
Question: What is photosynthesis?
Answer:
'''

#------- Basic usage (without training) -----------------
import dspy

# 1. Configure the language model
lm = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=lm)

# 2. Define a signature (input -> output specification)
class QuestionAnswer(dspy.Signature):
    """Answer questions with short factual answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# 3. Create a simple module
class BasicQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(QuestionAnswer)
    
    def forward(self, question):
        return self.generate_answer(question=question)

# 4. Use it
qa = BasicQA()
response = qa(question="What is the capital of France?")
print(response.answer)


#-------- With RAG -----------------------
import dspy

# Configure with a retriever
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2)

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)

# Use it
rag = RAG()
response = rag("What are the main causes of climate change?")
print(response.answer)

#----------- Training --------------------
from dspy.teleprompt import BootstrapFewShot

# Your training data
trainset = [
    dspy.Example(question="What is photosynthesis?", 
                 answer="The process by which plants convert light into energy").with_inputs("question"),
    # ... more examples
]

# Define a metric
def validate_answer(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# Compile/optimize the program
teleprompter = BootstrapFewShot(metric=validate_answer)
optimized_rag = teleprompter.compile(RAG(), trainset=trainset)

# Now use the optimized version
response = optimized_rag("What is photosynthesis?")

#---------- You can save the trained model as json ----------------
# After optimization
teleprompter = BootstrapFewShot(metric=validate_answer)
optimized_rag = teleprompter.compile(RAG(), trainset=trainset)

# Save it
optimized_rag.save("my_optimized_rag.json")

# Later, load it back
loaded_rag = RAG()
loaded_rag.load("my_optimized_rag.json")

# Use it directly - no need to compile/optimize again
response = loaded_rag("What is photosynthesis?")



