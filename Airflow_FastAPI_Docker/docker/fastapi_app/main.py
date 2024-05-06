from fastapi import FastAPI, HTTPException
from airflow.api.client.local_client import Client

#Define FastAPI app
app = FastAPI()
airflow_client = Client(api_base_url='http://localhost:8080')

#Define route to handle request
@app.post("/trigger_dag/{dag_id}")
async def trigger_dag(dag_id: str):
    try:
        airflow_client.trigger_dag(dag_id)
        return {"message": f"DAG '{dag_id}' triggered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))