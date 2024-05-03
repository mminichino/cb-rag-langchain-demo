# Couchbase Vector Demo 1.0.0

## Prerequisites
- OpenAI API key

Steps to setup and run the demo:
```
python3 -m venv rag_demo
```
```
. rag_demo/bin/activate
```
```
pip install git+https://github.com/mminichino/cb-rag-langchain-demo
```
```
demo_run
```
To exit from the Python virtual environment:
```
deactivate
```
NOTE: The reset feature to flush the bucket to start over is not supported on Capella. Please flush the bucket from the Capella UI.
<br>
![Rag Demo](https://raw.githubusercontent.com/mminichino/cb-rag-langchain-demo/main/doc/ragdemo.png)
