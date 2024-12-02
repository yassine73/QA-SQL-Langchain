from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from sys import argv

def ai_sql_executor(text_input):
    ## SQLite DB connection
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    llm = ChatOllama(model="llama2")

    ## Get SQL Schema
    def get_schema(_):
        return db.get_table_info()

    def run_query(test):
        return db.run(test)


    template = """
    Based on Table schema below. write a SQL query that would answer the user's question:
    {schema}

    Question: {question}
    SQL Query: 
    """

    prompt = ChatPromptTemplate.from_template(template)

    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema) ## Run function get_schema and then replace the result in {schema} inside prompt template 
        | prompt
        | llm.bind(stop=["\nSQL Result:"]) ## Stop the llm from generation text after SQL Result
        | StrOutputParser() ## To make sure that the output of this chain is string
    )

    ## Generate template that execute SQL Query in database and reading the executed output then generate natural language response 

    template = """
    Based on the table schema bellow, sql query, and sql response, write a natural lnaguage response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    run_query('SELECT COUNT(*) AS TotalArtists FROM Artist;')

    full_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=get_schema,
            response=lambda variable: run_query(variable["query"])
        )
        | prompt
        | llm
    )

    response = full_chain.invoke({"question" : text_input})
    return response