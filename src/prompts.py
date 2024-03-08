RAG_TEMPLATE = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
""".strip()

REWRITE_QUERY_TEMPLATE = """
Given a chat history and a new query from user, your task is to expand the query so that it is relevant and contains all information needed for searching without the history chat.
Expand and contextualize the query as best as you can in one or two short sentences.
Chat history:
{history}
User query: {query}
""".strip()