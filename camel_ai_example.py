assistant = ChatAgent(
    system_message="You are a senior software architect.",
    model=model,
)

reviewer = ChatAgent(
    system_message="You are a strict code reviewer.",
    model=model,
)

question = "Design a REST API for an insurance claim system."

architect_reply = assistant.step(question)

review_reply = reviewer.step(architect_reply.msg.content)

print(architect_reply.msg.content)
print(review_reply.msg.content)
