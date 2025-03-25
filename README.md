# ais_workshop

Code for Spring 2025 AIS Lab Undergrad Workshop 


## Setup 

Install dependencies

```bash
python -m venv .venv 

pip install -r requirements.txt
```

## Experiment tracking with Weight & Biases
Login
```bash
wandb login
```


## Workflow orchestration with Prefect

Start prefect server
```bash
prefect server start

prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Create work pool (For scheduling)
```bash
prefect work-pool create my-work-pool --type process

prefect worker start --pool my-work-pool
```


## Model deployment with fastapi

To start the server
```bash
uvicorn deployment.server:app --port 8000
```