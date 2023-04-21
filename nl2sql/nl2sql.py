"""Streamlit app generate SQL statements from natural language queries."""

import os
from typing import Any, Dict, Optional
from urllib.parse import quote_plus

import pandas as pd
import sqlalchemy as sa
import streamlit as st
from langchain import OpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    GPTSQLStructStoreIndex,
    LLMPredictor,
    ServiceContext,
    SQLDatabase,
)
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.common.struct_store.schema import SQLContextContainer
from llama_index.indices.struct_store import SQLContextContainerBuilder


# Function to create a connection string
def create_connection_string(host: str, port: int, dbname: str, user: str, password: str) -> str:
    """Create a connection string for the database."""
    quote_plus_password = quote_plus(password)
    quote_plus_user = quote_plus(user)
    return f"postgresql://{quote_plus_user}:{quote_plus_password }@{host}:{port}/{dbname}"


# Function to connect to the database
@st.cache_resource
def connect_to_database(connection_string: str) -> Optional[sa.engine.Connection]:
    """Connect to the Redshift/Postgres database."""
    try:
        return sa.create_engine(connection_string)
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None


@st.cache_resource
def create_llama_db_wrapper(
    _connection: sa.engine.Connection, connection_string: str
) -> SQLDatabase:
    """Create a SQLDatabase wrapper for the database.

    Args:
        _connection (sa.engine.Connection): connection
        connection_string (str): Placeholder to force cache invalidation

    Returns:
        SQLDatabase: SQLDatabase wrapper for the database
    """
    return SQLDatabase(_connection)


@st.cache_resource
def build_table_schema_index(
    _sql_database: SQLDatabase, connection_string: str, model_name: str
) -> tuple[Any, Any]:
    """Build a table schema index from the SQL database.

    Args:
        _sql_database (SQLDatabase): SQL database
        connection_string (str): Placeholder to force cache invalidation

    Returns:
        tuple[Any, Any]: table_schema_index, context_builder
    """
    # noop to avoid unused variable warning and autoprint to streamlit
    connection_string = connection_string

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=model_name))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # build a vector index from the table schema information
    context_builder = SQLContextContainerBuilder(_sql_database)
    table_schema_index = context_builder.derive_index_from_context(
        GPTSimpleVectorIndex,
        service_context=service_context,
    )

    return table_schema_index, context_builder


@st.cache_resource
def build_sql_context_container(
    _context_builder: SQLContextContainerBuilder, _table_schema_index: BaseGPTIndex, query_str: str
) -> SQLContextContainer:
    """Build a SQL context container from the table schema index."""
    _context_builder.query_index_for_context(
        _table_schema_index, query_str, store_context_str=True, similarity_top_k=2
    )

    return _context_builder.build_context_container()


@st.cache_resource
def create_sql_struct_store_index(
    _sql_database: SQLDatabase, connection_string: str
) -> GPTSQLStructStoreIndex:
    """Create a SQL structure index from the SQL database.

    Args:
        _sql_database (SQLDatabase): SQL database
        connection_string (str): Placeholder to force cache invalidation

    Returns:
        GPTSQLStructStoreIndex: SQL structure index
    """
    # noop to avoid unused variable warning and autoprint to streamlit
    connection_string = connection_string

    return GPTSQLStructStoreIndex.from_documents([], sql_database=_sql_database)


@st.cache_data(ttl=60)
def query_sql_structure_store(
    _index: GPTSQLStructStoreIndex,
    _sql_context_container: SQLContextContainer,
    connection_string: str,
    query_str: str,
) -> Dict[str, Any]:
    """Query the SQL structure index.

    Args:
        _index (GPTSQLStructStoreIndex): SQL structure index
        _sql_context_container (SQLContextContainer): SQL context container
        connection_string (str): Placeholder to force cache invalidation
        query_str (str): SQL query string

    Returns:
        Dict[str, Any]: Query response
    """
    # noop to avoid unused variable warning and autoprint to streamlit
    connection_string = connection_string

    response = _index.query(query_str, sql_context_container=_sql_context_container)

    return response


def main() -> None:
    """Start the streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Natural Language to SQL Query Executor")

    # Left pane for Redshift connection input controls
    with st.sidebar:
        st.header("Connect to Redshift/Postgres")
        db_credentials = st.secrets.get("db_credentials", {})
        host = st.text_input("Host", value=db_credentials.get("host", ""))
        port = st.number_input(
            "Port", min_value=1, max_value=65535, value=db_credentials.get("port", 5439)
        )
        dbname = st.text_input("Database name", value=db_credentials.get("database", ""))
        user = st.text_input("Username", value=db_credentials.get("username", ""))
        password = st.text_input(
            "Password", type="password", value=db_credentials.get("password", "")
        )

        connect_button = st.button("Connect")

    # Connect to DB when 'Connect' is clicked
    if connect_button or st.session_state.get("connect_clicked"):
        st.session_state.connect_clicked = True
        # change label of connect button to 'Connected'
        connection_string = create_connection_string(host, port, dbname, user, password)

        connection = connect_to_database(connection_string=connection_string)

        # Right pane for SQL query input and execution
        if connection:
            open_api_key = st.text_input(
                "OpenAI API Key",
                value=st.secrets.get("open_api_key", st.session_state.get("open_api_key", "")),
                type="password",
            )

            btn_open_api_key = st.button("Enter")

            if open_api_key or btn_open_api_key:
                session_openapi_key = st.session_state.get("open_api_key")

                # keep streamlit state that open_api_key had been entered
                if not session_openapi_key or session_openapi_key != open_api_key:
                    st.session_state.open_api_key = open_api_key

                # openai libs access the key via this environment variable
                os.environ["OPENAI_API_KEY"] = st.session_state.open_api_key

                model_name = st.selectbox(
                    "Choose OpenAI model", ("none", "gpt-3.5-turbo", "text-davinci-003", "gpt-4")
                )

                if model_name != "none":
                    # Create LLama DB wrapper
                    st.markdown(
                        (
                            ":blue[Create DB wrapper."
                            f" Inspect schemas, tables and views inside **_{dbname}_** DB.]"
                        )
                    )
                    sql_database = create_llama_db_wrapper(
                        connection, connection_string=connection_string
                    )

                    # build llama sqlindex
                    st.write(":blue[Build table schema index.]")
                    table_schema_index, context_builder = build_table_schema_index(
                        sql_database, connection_string=connection_string, model_name=model_name
                    )

                    query_str = st.text_area("Enter your NL query here:")
                    run_button = st.button("Run")

                    # Execute the SQL query when 'Run' button is clicked
                    if run_button or query_str:
                        index = create_sql_struct_store_index(
                            sql_database, connection_string=connection_string
                        )

                        sql_context_container = build_sql_context_container(
                            context_builder, table_schema_index, query_str
                        )

                        # st.write(sql_context_container.context_str)
                        # st.write(sql_context_container.context_dict)

                        st.markdown(":blue[Prepare and execute query...]")
                        try:
                            response = query_sql_structure_store(
                                _index=index,
                                _sql_context_container=sql_context_container,
                                connection_string=connection_string,
                                query_str=query_str,
                            )
                        except Exception as ex:
                            print(f"Exception type:{type(ex)}")
                            print(ex)
                            st.markdown(
                                ":red[We couldn't generate a valid SQL query. "
                                "Please try to refine your question with schema, "
                                "table or column names.]"
                            )
                            return

                        st.markdown(
                            f":blue[Generated query:] :green[_{response.extra_info['sql_query']}_]"
                        )
                        st.dataframe(pd.DataFrame(response.extra_info["result"]))


# Run the Streamlit app
if __name__ == "__main__":
    main()
