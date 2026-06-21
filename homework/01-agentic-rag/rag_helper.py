import os

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {'question': 3.0, 'section': 0.5}
        filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer





class HomeworkRAG(RAGBase):
    def __init__(self, index, llm_client, **kwargs):
        kwargs.setdefault('model', 'models/gemini-3.1-flash-lite')  
        super().__init__(index, llm_client, **kwargs)

    def search(self, query, num_results=5):
        #boost_dict = {'filename': 2.0, 'content': 1.0}
        
        return self.index.search(
            query,
            num_results=num_results,
            #boost_dict=boost_dict
        )

    def build_context(self, search_results):
        """Builds a context string from the search results, including both the filename and content of each document."""
        lines = []
        for doc in search_results:
            lines.append(f"File: {doc['filename']}")
            lines.append(doc['content'])
            lines.append('')
        return '\n'.join(lines).strip()

    def llm(self, prompt):
        """Override the llm method to use the chat completions endpoint."""
        input_messages = [
            {'role': 'developer', 'content': self.instructions}, 
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages
        )

        return response

    def rag(self, query):
        """Updated RAG method that returns both the answer and the number of prompt tokens used."""
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        
        response = self.llm(prompt)
        
        answer = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        
        return answer, prompt_tokens
