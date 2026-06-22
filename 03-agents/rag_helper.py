from ingest import  load_faq_data
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
        model='models/gemini-3.1-flash-lite'
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

    # The context представляет собой отформатированную строку со всеми результатами поиска:
    def build_context(self,search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')

        return '\n'.join(lines).strip()

    # Каждый документ становится блоком с разделом, вопросом и ответом. Этот формат облегчает чтение LLM. 
    # Мы только что повернули словарь в строку — ничего особенного.

    def build_prompt(self, question, search_results):
        """ Build a prompt for the LLM by combining the question and search results."""
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(
            question=question,
            context=context
        )
        return prompt.strip() # убираем лишние пробелы в начале и конце, если они есть

    def llm(self,user_prompt):
        # Заменяем 'developer' на 'system', так как шлюз Gemini ожидает именно эту роль
        message_history = [
            {'role': 'system', 'content': self.instructions},
            {'role': 'user', 'content': user_prompt}
        ]

        # Используем стандартный интерфейс SDK
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages= message_history,
            temperature=0.0 # Фиксируем температуру на 0 для предсказуемости RAG
        )

        # Достаем текст из стандартизированного объекта ответа
        return response.choices[0].message.content

    def rag(self, query):

        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        
        return answer


class OllamaRAG(RAGBase):
    # В этом классе мы переопределяем метод llm, чтобы использовать Ollama вместо OpenAI SDK.
    def llm(self, user_prompt):
        response = self.llm_client.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': self.instructions},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=0.0
        )
        return response['choices'][0]['message']['content']
    

class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {'course': self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )