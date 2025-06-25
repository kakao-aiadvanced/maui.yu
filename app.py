# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

from pprint import pprint
from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tavily import TavilyClient

# 환경 변수 로드
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        web_search: whether to add search
        web_search_count: number of web search
        generation: LLM generation
        generation_count: number of generations
        documents: list of documents
    """

    question: str
    web_search: str
    web_search_count: int
    generation: str
    generation_count: int
    documents: List[Document]

def build_message(value):
    web_search_limit_exceeded = value.get("web_search_count", 0) > 1
    generation_limit_exceeded = value.get("generation_count", 0) > 1
    if web_search_limit_exceeded:
        return "failed: not relevant"
    elif generation_limit_exceeded:
        return "failed: hallucination"
    else:
        # 문서 출처를 마크다운 리스트로 생성
        md_lines = []
        seen_sources = set()
        for doc in value.get("documents", []):
            pprint(doc.metadata)
            title = doc.metadata.get("title", "")
            source = doc.metadata.get("source", "")
            # 중복 source 필터링
            if source and source in seen_sources:
                continue
            if source:
                seen_sources.add(source)

            if title and source:
                md_lines.append(f"- [{title}]({source})")
            elif title:
                md_lines.append(f"- {title}")
            elif source:
                md_lines.append(f"- {source}")
        md_result = "\n".join(md_lines)
        return value["generation"] + "\n\n출처: \n" + md_result

# URL 처리 함수
@st.cache_resource
def process_urls():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    return text_splitter.split_documents(docs_list)

# 벡터 스토어 초기화
@st.cache_resource
def initialize_vectorstore():
    chunks = process_urls()
    return Chroma.from_documents(
        documents=chunks,
        collection_name="rag-chroma",
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    )

# 체인 초기화
@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    ### Router
    system = """You are an expert at routing a user question to a vectorstore or web search.
    Use the vectorstore for questions on LLM agents, prompt, prompt engineering, and adversarial attacks.
    You do not need to be stringent with the keywords in the question related to these topics.
    Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
    Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}"),
        ]
    )

    question_router = prompt | llm | JsonOutputParser()

    # question = "llm agent memory"
    question = "What is prompt?"
    # Test retriever before using
    try:
        docs = retriever.invoke(question)
        print(question_router.invoke({"question": question}))
    except Exception as e:
        print(f"Error during initialization: {e}")

    ### Retrieval Grader
    system = """You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n document: {document} "),
        ]
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    question = "What is prompt?"
    try:
        docs = retriever.invoke(question)
        doc_txt = docs[0].page_content
        print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
    except Exception as e:
        print(f"Error during retrieval grader test: {e}")

    ### Generate
    system = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    question = "What is prompt?"
    try:
        docs = retriever.invoke(question)
        generation = rag_chain.invoke({"context": docs, "question": question})
        print(generation)
    except Exception as e:
        print(f"Error during RAG chain test: {e}")

    ### Hallucination Grader
    system = """You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    try:
        hallucination_grader.invoke({"documents": docs, "generation": generation})
    except Exception as e:
        print(f"Error during hallucination grader test: {e}")

    ### Answer Grader
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    # Prompt
    system = """You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n answer: {generation} "),
        ]
    )

    answer_grader = prompt | llm | JsonOutputParser()
    try:
        answer_grader.invoke({"question": question, "generation": generation})
    except Exception as e:
        print(f"Error during answer grader test: {e}")

    ### Nodes

    def retrieve(state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)
        print(question)
        print(documents)
        return {"documents": documents, "question": question}


    def generate(state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        generation_count = state.get("generation_count", 0) + 1
        return {"documents": documents, "question": question, "generation": generation, "generation_count": generation_count}


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        web_search_count = state.get("web_search_count", 0)

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: ALL DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                web_search = "Yes"
                return {
                    "documents": filtered_docs,
                    "question": question,
                    "web_search": web_search,
                    "web_search_count": web_search_count,
                }

        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search, 
            "web_search_count": web_search_count,
        }


    def web_search(state):
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        print("---WEB SEARCH---")
        print(state)
        question = state["question"]
        documents = None
        if "documents" in state:
            documents = state["documents"]

        # Web search
        docs = tavily.search(query=question)['results']
    #    [{'title': 'Where will Lionel Messi play in 2024? Cities, stadiums Inter Miami ...', 'url': 'https://www.sportingnews.com/us/soccer/news/where-lionel-messi-play-2024-inter-miami-cities-road-schedule/23334c5768cebee9021e71d0', 'content': "Here is how Inter Miami's road schedule will look for the coming regular season:\nInter Miami home stadium for 2024 MLS season\nFor their home matches through the 2024 campaign, Inter Miami will once again play at\xa0DRV PNK Stadium in Fort Lauderdale, Florida.\n Cities, stadiums Inter Miami visit on road MLS schedule for new season\nWith Lionel Messi set to embark on his first full season with Inter Miami, fans across the United States will be clamoring to see when the Argentine superstar will visit their city in 2024.\n MLS Season Pass is separate from Apple TV+, meaning those with Apple TV+ would still need an MLS Season Pass subscription to access the complete slate of games, while those without Apple TV+ can still sign up for MLS Season Pass without needing a full Apple TV+ subscription.\n SUBSCRIBE TO MLS SEASON PASS NOW\nApple TV is the official home of the MLS regular season and playoffs, with every match for every team available to stream around the world with no blackouts. How to watch Inter Miami in 2024 MLS season\nLast season, Major League Soccer kicked off a 10-year broadcast rights deal with Apple that sees every single match for the next decade streamed exclusively on Apple's streaming platform.\n", 'score': 0.98612, 'raw_content': None}, {'title': 'Is Lionel Messi playing today? Status for next Inter Miami game in 2024 ...', 'url': 'https://www.sportingnews.com/us/soccer/news/lionel-messi-playing-today-inter-miami-game-2024/129c2c378fee4d1f0102aa9d', 'content': '* Lionel Messi did not participate. Inter Miami schedule for Leagues Cup. The 2024 Leagues Cup is scheduled to begin on July 26, running for a month while the MLS season pauses play.. The final ...', 'score': 0.98209, 'raw_content': None}, {'title': 'Lionel Messi joins Inter Miami: Schedule, MLS tickets to see him play', 'url': 'https://www.usatoday.com/story/sports/mls/2023/06/07/lionel-messi-inter-miami-schedule-tickets/70299298007/', 'content': 'Lionel Messi joins Inter Miami: Full schedule, MLS tickets to see Messi play in US\nLionel Messi\xa0is taking his talents to South Beach.\nMessi,\xa0the 2022 World Cup champion, announced on Wednesday that he will join Major League Soccer\'s Inter Miami CF, a pro soccer club owned by David Beckham, after exiting Ligue 1\'s Paris Saint-Germain following two seasons.\n Tickets to Inter Miami\'s game on June 10 range from $40-$55, but the price tag to see Inter Miami play LigaMX\'s Cruz Azul on July 21 soared to $495 in anticipation of what\'s expected to be Messi\'s first home game, TicketSmarter CEO Jeff Goodman told USA TODAY Sports.\n Each team will play a minimum of two games in the group stage, similar to the World Cup format, with the possibility of more games if the team advances to the knockout rounds.\n "\nAccording to Goodman, nearly 2,000 Inter Miami tickets sold on TicketSmarter the day of Messi\'s announcement Wednesday, compared to under 50 tickets being sold on the platform over the weekend.\n If the Barcelona thing didn\'t work out, I wanted to leave Europe, get out of the spotlight and think more of my family.', 'score': 0.97895, 'raw_content': None}, {'title': "Lionel Messi's 2023 Inter Miami schedule: Every match in MLS, Leagues ...", 'url': 'https://www.sportingnews.com/us/soccer/news/lionel-messi-2023-inter-miami-schedule/d3buao2mhfp7uculkdz3nsc4', 'content': "MORE:\xa0Trophies that Lionel Messi can win with Inter Miami in USA\nLeagues Cup\nIn his first three matches with Inter Miami, Lionel Messi lifted the club into the Leagues Cup Round of 16 thanks to three straight home wins that he helped orchestrate.\n Edition\nLionel Messi's 2023 Inter Miami schedule: Every match in MLS, Leagues Cup and U.S. Open Cup\nLionel Messi is taking North America by storm after scoring in his first three matches for his new club Inter Miami CF.\n MORE: Messi's Miami apartment | Messi's wife & family | Messi's net worth\nLionel Messi, Inter Miami 2023 schedule\nBelow are the remaining games for Inter Miami that Messi will be a part of. MLS\nAfter the Leagues Cup is out of the way, Inter Miami will have 12 MLS matchdays left in a bid to reach the MLS Cup playoffs.\n Inter Miami can still make MLS playoffs\xa0with Lionel Messi\nU.S. Open Cup\nInter Miami reached the semifinal of the competition before Messi and friends joined.", 'score': 0.97298, 'raw_content': None}, {'title': 'Messi, Argentina to play in Chicago, DC before Copa America: More info', 'url': 'https://www.usatoday.com/story/sports/soccer/2024/05/20/messi-argentina-to-play-in-chicago-dc-before-copa-america-more-info/73770204007/', 'content': "1:00. World Cup champion Lionel Messi will participate in two Argentina friendlies early next month before Copa América begins June 20. Messi was officially named to Argentina's 29-man roster ...", 'score': 0.97096, 'raw_content': None}]
    #
    #

        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(
            page_content=web_results,
            metadata={
                "title": docs[0]["title"] if "title" in docs[0] else "",
                "source": docs[0]["url"] if "url" in docs[0] else ""
            }
        )
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]

        web_search_count = state.get("web_search_count", 0) + 1

        return {"documents": documents, "question": question, "web_search_count": web_search_count}


    ### Edges


    def route_question(state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        print(question)
        source = question_router.invoke({"question": question})
        print(source)
        print(source["datasource"])
        if source["datasource"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source["datasource"] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"


    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]
        web_search_count = state.get("web_search_count", 0)

        if web_search == "Yes":
            if web_search_count > 1:
                print(f"---DECISION: WEB SEARCH LIMIT EXCEEDED. WEB SERCH COUNT: {state['web_search_count']}---")
                print("failed: not relevant")
                return "relevance check failed"
            
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"


    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        generation_count = state.get("generation_count", 0)

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":

            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            
            # "할루시네이션 실패" 의도적으로 발생 시키기
            # if generation_count > 1:
            #     print(f"---DECISION: GENERATION LIMIT EXCEEDED. GENERATION COUNT: {state['generation_count']}---")
            #     print("failed: hallucination")
            #     return "hallucination check failed"
            # return "not supported" # TODO: 삭제
            
            return "useful"
        else:
            if generation_count > 1:
                print(f"---DECISION: GENERATION LIMIT EXCEEDED. GENERATION COUNT: {state['generation_count']}---")
                print("failed: hallucination")
                return "hallucination check failed"

            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"


    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae

    # Graph Build
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
            "relevance check failed": END
        },
    )
    workflow.add_edge("websearch", "grade_documents")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "hallucination check failed": END,
            "useful": END,
            # "not useful": "websearch",
        },
    )

    return workflow.compile()

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG Agent", page_icon="🤖")
    st.title("RAG Agent")
    st.caption("RAG Agent with LangGraph and Tavily")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 체인 초기화
        chain = initialize_chain()

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                # 초기 state 설정
                inputs = {
                    "question": prompt,
                    "web_search": "No",
                    "web_search_count": 0,
                    "generation": "",
                    "generation_count": 0,
                    "documents": []
                }
                
                # 워크플로우 실행
                final_state = None
                for output in chain.stream(inputs):
                    for key, value in output.items():
                        print(f"Finished running: {key}")
                        final_state = value
                
                # 최종 응답 표시
                if final_state:
                    response_text = build_message(final_state)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    error_msg = "응답 생성 중 오류가 발생했습니다."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
