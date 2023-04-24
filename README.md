
# Natural Language to SQL Query Generator(nl2sql)
Streamlit App for natural language to SQL query generation. Tested with Postgres and Redshift connections but any SQLAlchemy compatible database should work. You need an OpenAI API key to use this app. You can get one [here](https://platform.openai.com/)

- [Natural Language to SQL Query Generator(nl2sql)](#natural-language-to-sql-query-generatornl2sql)
  - [💻Example Usage](#example-usage)
    - [Start on localhost:](#start-on-localhost)
  - [🔧Dependencies](#dependencies)

## 💻Example Usage

TLDR: Click on the [link](https://dimitar-petrunov-sagedata-nl2sql-nl2sqlnl2sql-epjv90.streamlit.app/) to see it in action.

### Start on localhost:

Requires python >= 3.9

```
pip install -r requirements.txt

streamlit run n2lsql/nl2sql.py
```

Navigate to your streamlit app at http://localhost:8501


1. Input database connection details.
2. Enter your OpenAI Api key.
3. Select the GPT model from OpenAI to use.
4. Input your natural language query and click 'Run'
5. If you don't get the desired results then be precise about the tables and columns that contain the requested data in your natural language query.
6. The natural language query is converted to a SQL query and the results are displayed in a table.




![](docs/img/nl2sql.png)


## 🔧Dependencies
You need an OpenAI API key to use this app. You can get one [here](https://platform.openai.com/)

Based on the awsome [llama-index](https://github.com/jerryjliu/llama_index) project. Uses [Streamlit](https://streamlit.io/) for the UI.
