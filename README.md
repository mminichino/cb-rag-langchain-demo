# Couchbase Vector Demo 1.0.1

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
Enter the parameters needed to connect to your Couchbase server. You will need a Capella API key to create the bucket on Capella. Otherwise, if you are using Capella, create the bucket in the Capella UI before you run the demo.
<br>
Click Configure to configure the cluster for the demo. Then click Start Demo to start the demo.
![Rag Demo](https://raw.githubusercontent.com/mminichino/cb-rag-langchain-demo/main/doc/ragdemo.png)
